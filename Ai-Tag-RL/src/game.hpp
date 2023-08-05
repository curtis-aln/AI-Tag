#pragma once

#include "utility.hpp"
#include "Agent.hpp"
#include "settings.hpp"

/* This is a Game which will exist as */
class Game : GameSettings
{
public:
	int timeRemaining = gameFrameLength;
	std::vector<Agent> agents{};
	NeuralNetwork networks[agentsPergame] = {};


public:
	Game() { agents.reserve(agentsPergame); }

	void initAgent(const Agent& agent) { agents.emplace_back(agent); }

	void initiliseGame(const std::vector<sf::Vector2f>& starting_positions)
	{
		// other re-settings
		timeRemaining = gameFrameLength;

		for (unsigned agent = 0; agent < agentsPergame; agent++)
		{
			agents[agent].reset();
		}

		agents[0].tagged = true;
		agents[0].position = starting_positions[0];
		agents[1].position = starting_positions[1];
	}

	bool tick()
	{
		for (unsigned i = 0; i < agentsPergame; i++)
		{
			agents[i].update(networks[i], agents);
		}
		return --timeRemaining == 0;
	}
};