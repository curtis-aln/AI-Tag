#pragma once

#include "SFML/Graphics.hpp"
#include "utility.hpp"
#include "settings.hpp"
#include "NeuralNetwork.hpp"

#include <cmath>


class Agent : AgentSettings
{
public:
	explicit Agent(const sf::Vector2f& Position) : position(Position) {}

	// hard reset all of the agent's information for another game
	void reset()
	{
		aliveTime = 0; network_score = 0; m_tagCooldown = 0; tagged = false;
		m_velocity = { 0.f, 0.f }; position = randPointOutCircle(Settings::bounds);
	}


	void update(NeuralNetwork& network, std::vector<Agent>& agents)
	{
		// computing the velocity from the neural network
		setNetworkInputs(network, agents);
		network.compute_output();
		position += { network.outputs[0] * 5, network.outputs[1] * 5};

		// preventing overlap with the game border or other agent(s)
		agentCollisions(agents);
		border(Settings::bounds, position, radius);

		aliveTime++;

		network_score += tagged;

		if (tagged && m_tagCooldown > 0)
		{
			m_tagCooldown--;
		}
	}


private:
	void setNetworkInputs(NeuralNetwork& network, std::vector<Agent>& agents) const
	{
		// the agent's personal information comes first so it does not get confused
		sf::Vector2f relativeBounds = relativePosToCircle(Settings::bounds, position);
		network.inputs[0] = relativeBounds.x;          // position Y border
		network.inputs[1] = relativeBounds.y;          // position Y border
		network.inputs[2] = m_velocity.x / maxSpeed;   // velocity X
		network.inputs[3] = m_velocity.y / maxSpeed;   // velocity Y
		network.inputs[4] = (tagged == 1) ? 1.f : -1.f;

		// other agent information is separated
		unsigned index = 4;
		for (Agent& agent : agents)
		{
			if (agent.position == position) continue;

			relativeBounds = relativePosToCircle(Settings::bounds, agent.position);
			network.inputs[++index] = relativeBounds.x;               // position X
			network.inputs[++index] = relativeBounds.y;               // position Y
			network.inputs[++index] = agent.m_velocity.x / maxSpeed;  // velocity X
			network.inputs[++index] = agent.m_velocity.y / maxSpeed;  // velocity Y
			network.inputs[++index] = (agent.tagged == 1) ? 1.f : -1.f;
		}
	}


	void agentCollisions(std::vector<Agent>& agents)
	{
		for (Agent& agent : agents)
		{
			if (agent.position != position)
			{
				agentCollision(&agent);

				const float diam = (Settings::bounds.radius - radius) * 2;
				const float distNorm = distSquared(position, agent.position) / (diam * diam);
				if (tagged) network_score += distNorm + 0.5f;
				//else network_score += abs((1.f - distNorm) - 0.5f);
			}
		}
	}


	void tagLogistics(Agent* agent)
	{
		if (tagged && m_tagCooldown <= 0 && aliveTime > GameSettings::gameStartImmunity)
		{
			agent->tagged = true;
			//agent->network_score += taggedPenalty;
			//network_score -= taggedPenalty;
			agent->m_tagCooldown = tagcooldownamount;
			tagged = false;
		}
	}


	bool agentCollision(Agent* agent)
	{

		const sf::Vector2f relative_position = agent->position - position;
		const float dist_squared = distSquared(this->position, agent->position);
		constexpr float sum_radii = radius + radius;

		if (dist_squared > sum_radii * sum_radii || dist_squared < 0)
			return false;

		tagLogistics(agent);

		const float dist = std::sqrt(dist_squared);
		const sf::Vector2f normal_vector = relative_position / dist;
		const sf::Vector2f correction = (sum_radii - dist) * normal_vector * 0.5f;

		// Move the agents to prevent them from interpenetrating
		const sf::Vector2f displacement = correction * (radius / sum_radii);
		position -= displacement;
		agent->position += displacement;

		return true;
	}


	void applyFriction(const float strength) { m_velocity /= strength; }


public:
	sf::Vector2f gameStartPos{};
	sf::Vector2f position{};
	sf::Vector2f m_velocity{};
	sf::Vector2f m_accelaration{};

	float network_score = 0;
	bool tagged = false;

private:
	unsigned m_tagCooldown = 0;
	unsigned aliveTime = 0;
};