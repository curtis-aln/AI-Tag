#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
#include <chrono>

#include "../settings.hpp"
#include "../utility.hpp"
#include "../Agent.hpp"
#include "../game.hpp"


struct BestNetworkInfo
{
	float score;
	sf::Vector2f learnerPosition;
	sf::Vector2f trainerPosition;
	NeuralNetwork* Network;
};



class Simulation : Settings, DeltaTime
{
	// ---------- SFML window ---------- //
	sf::Clock m_clock{};
	sf::RenderWindow m_window{ sf::VideoMode(
		static_cast<unsigned>(windowSize.x), static_cast<unsigned>(windowSize.y)), simulationName };

	// ---------- containers ---------- //
	std::vector<Game> m_allGames; // run in parrelel (multi-threading)

	// ---------- other ---------- //
	BetterFrameRates<60> m_frameRateManager;
	ReinforcementLearning selfRL{};
	BestNetworkInfo best_net_info{};

	// ---------- statistics ---------- //
	bool m_closeSim  = false;
	bool m_paused    = false;
	bool m_debug     = true;
	bool m_rendering = true;
	bool m_debugValue= false;
	bool m_allrender = false;
	bool m_auto_save = false;
	bool fastForward = false;

	unsigned m_totalFrameCount = 0;
	unsigned m_generationCount = 1;
	double m_totalRunTime      = 0;

	// ---------- debugging ---------- //
	sf::CircleShape m_agentRenderCircle{};
	sf::CircleShape m_gameBorderRenderer{};

	FontManager scores;


	NeuralNetwork trainerAgentNet{};


public:
	explicit Simulation();
	static void printNetworkInfo();
	static void runGame(Game* game);
	void run();
	void prepareNextAgents();
	void uihandeling();
	void tickGames(bool& stop);
	void endOfGenStats();
	void resetGames();
	void getTopNet();

	void endFrame();
	void initGames();
	void saveNetworkData();
	void loadNetworkData();

	void pollEvents();
	void keyPressEvents(const sf::Keyboard::Key& event_key_code);
	static sf::Color getColor(bool tagged);
	void renderAgents();
	void debugAgents();

	// ---------- graphics ---------- //
	void renderFrame();
	void setWindowTitle();
	void debugAgent(const Agent* agent);
	void initDebugGraphics();

};