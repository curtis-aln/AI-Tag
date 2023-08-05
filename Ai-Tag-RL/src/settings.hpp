#pragma once

#include <SFML/Graphics.hpp>
#include "utility.hpp"

struct Settings
{
	static constexpr unsigned parrelelGames      = 100;

	static constexpr unsigned frameRate          = 800;
	static constexpr unsigned bufferCirclePoints = 20;
	static constexpr unsigned alignmentFreq      = 30'000;
	static constexpr unsigned autoSaveFreq       = 250;


	inline static const sf::Vector2f   windowSize    = { 800, 800 };
	inline static const sf::Color      windowColor   = { 20, 20, 20 };
	inline static const CircularBorder bounds{{windowSize.x/2, windowSize.y/2}, 350.f};

	inline static const std::string simulationName = "TAG AI Sim";
	inline static const std::string saveFileName = "data/data.json";

	inline static std::vector<sf::Color> colors = {
		{0, 90, 255, 255},// blue
		{0, 255, 100, 255},  // green
	};
};


struct GameSettings
{
	static constexpr unsigned agentsPergame     = 2;
	static constexpr unsigned gameFrameLength   = 2000;
	static constexpr unsigned gameStartImmunity = 50;
};


struct AgentSettings
{
	inline static const sf::Color notItColor = { 50, 50, 50, 255 };
	inline static const sf::Color itColor    = { 255, 0, 0, 255 };

	static constexpr float friction = 1.00f;
	static constexpr float maxSpeed = 16.50f;
	static constexpr float radius   = 30.0f;

	static constexpr unsigned tagcooldownamount = 50;

	static constexpr float borderTouchPenalty = 0.050f;
	static constexpr float lowSpeedPenalty    = 0.050f;
	static constexpr float taggedPenalty      = 400.00f;
};


struct NetSettings
{
	static constexpr unsigned NetworkLayers =4;
	static constexpr unsigned NN_dims[NetworkLayers] = {GameSettings::agentsPergame * 5, 18, 18, 2 };

	static constexpr unsigned largestnonInpLayer = NN_dims[1];
	static constexpr unsigned largestLayer = NN_dims[1];

	inline static constexpr float weight_mutation_rate = 0.5f;
	inline static constexpr float weight_mutation_range= 0.5f;

	inline static constexpr float bias_mutation_rate  = 0.5f;
	inline static constexpr float bias_mutation_range = 0.5f;

};