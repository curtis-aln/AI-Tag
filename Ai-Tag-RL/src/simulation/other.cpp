#include "simulation.hpp"
#include <nlohmann/json.hpp>

Simulation::Simulation() : DeltaTime(), scores(&m_window, 15)
{
	m_window.setFramerateLimit((m_rendering == true) ? 100 : 999'999);

	initGames();
	initDebugGraphics();
	printNetworkInfo();
}


void Simulation::printNetworkInfo()
{
	unsigned neurons = 0;
	unsigned weights = 0;

	for (unsigned i = 0; i < NetSettings::NetworkLayers; i++)
	{
		neurons += NetSettings::NN_dims[i];

		if (i < NetSettings::NetworkLayers - 1)
			weights += NetSettings::NN_dims[i] * NetSettings::NN_dims[i + 1];
	}

	std::cout << "Neural Network Config: \n";
	std::cout << "Layers:  " << NetSettings::NetworkLayers << "\n";
	std::cout << "Neurons: " << neurons << "\n";
	std::cout << "Biases:  " << neurons << "\n";
	std::cout << "weights: " << weights << "\n";
	std::cout << "params:  " << weights + neurons << "\n";
}


// creating the SFML shapes that will show debug information on the screen. the rest is handled by the vertex buffer
void Simulation::initDebugGraphics()
{
	// render circle
	constexpr float outline = 3;
	m_agentRenderCircle.setRadius(AgentSettings::radius - outline/2);
	m_agentRenderCircle.setOutlineThickness(outline);
	m_agentRenderCircle.setFillColor({ 255, 255, 255, 255 });

	// circular border
	m_gameBorderRenderer.setPosition(bounds.position - sf::Vector2f{ bounds.radius, bounds.radius });
	m_gameBorderRenderer.setRadius(bounds.radius);
	m_gameBorderRenderer.setOutlineColor({ 255, 255, 255, 255 });
	m_gameBorderRenderer.setFillColor({ 0, 0, 0, 0 });
	m_gameBorderRenderer.setOutlineThickness(3.f);
}


void Simulation::initGames()
{
	m_allGames.reserve(parrelelGames);

	for (unsigned i = 0; i < parrelelGames; i++)
	{
		Game game{};
		for (unsigned j = 0; j < GameSettings::agentsPergame; j++)
		{
			game.initAgent(Agent{ randPointInCircle(bounds) });
		}
		m_allGames.push_back(game);
	}
	std::cout << "[notice]: "<< m_allGames.size() << " games created" << "\n";
}


void Simulation:: saveNetworkData()
{
	std::cout << "[Notice]: Saving. . .\n";

	nlohmann::json agent_networks = {};
	for (unsigned i = 0; i < GameSettings::agentsPergame; i++)
	{
		for (NeuralNetwork& network : selfRL.policy)
			network.jsonFormat(agent_networks);
	}

	const nlohmann::json data = {
		{"gen", m_generationCount},
		{"time", m_totalRunTime},
		{"nets", agent_networks}
	};

	std::ofstream ofs("network_data.json");
	ofs << data.dump(3);
	ofs.close();
}

void Simulation::loadNetworkData()
{
	// reading data from file
	nlohmann::json simulationData = loadJsonData("network_data.json");
	m_generationCount = simulationData["gen"];
	m_totalRunTime = simulationData["time"];

	// shrinking variable names
	constexpr unsigned total_nets = ReinforcementLearning::snapshot_window;

	selfRL.reset_information();

	for (unsigned network_i = 0; network_i < total_nets; network_i++)
	{
		std::vector<std::vector<std::vector<float>>> agentWeights = simulationData["nets"][network_i]["weights"];
		std::vector<std::vector<float>> agentBiases = simulationData["nets"][network_i]["biases"];

		for (unsigned layer = 0; layer < NetSettings::NetworkLayers - 1; ++layer) // each network layer
		{
			for (unsigned node = 0; node < NetSettings::NN_dims[layer + 1]; ++node)
			{
				for (unsigned weight = 0; weight < NetSettings::NN_dims[layer]; ++weight)
				{
					selfRL.policy[selfRL.current_index].weights[layer][node][weight] = agentWeights[layer][node][weight];
				}
			}

			for (unsigned bias = 0; bias < NetSettings::NN_dims[layer + 1]; ++bias)
			{
				selfRL.policy[selfRL.current_index].biases[layer][bias] = agentBiases[layer][bias];
			}
		}

		selfRL.increment();
	}
	prepareNextAgents();
}