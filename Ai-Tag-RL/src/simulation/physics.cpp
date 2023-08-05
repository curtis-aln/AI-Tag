#include "simulation.hpp"


void Simulation::runGame(Game* game)
{
	for (unsigned i = 0; i < GameSettings::gameFrameLength; i++)
	{
		game->tick();
	}
}


void Simulation::uihandeling()
{
	for (unsigned i = 0; i < GameSettings::gameFrameLength; i++)
	{
		pollEvents();
		endFrame();
		setWindowTitle();

		if (m_rendering == true)
			renderFrame();
	}
}


void Simulation::run()
{
	while (!m_closeSim)
	{
		resetGames();
		bool stop = false;
		while (!stop && !m_closeSim)
		{
			if (!m_paused)
			{
				for (Game& game : m_allGames)
				{
					stop = game.tick();
				}
			}

			if (fastForward) m_rendering = false;


			if (m_rendering || (!m_rendering && m_totalFrameCount % 2000 == 0))
			{
				pollEvents();
				endFrame();
				setWindowTitle();
			}

			if (m_rendering == true)
				renderFrame();

			++m_totalFrameCount;
		}
		if (fastForward) { fastForward = false; m_rendering = true; }
		prepareNextAgents();
		endOfGenStats();
	}
}


void Simulation::prepareNextAgents()
{
	getTopNet();
	const NeuralNetwork* bestNetwork = best_net_info.Network;
	selfRL.add_neural_network(*bestNetwork, m_generationCount);
	

	// finding the next neural network to use for the teacher agent to train the learning agent
	const NeuralNetwork* newPastNetwork = selfRL.get_network(m_generationCount);

	for (Game& game : m_allGames)
	{
		bestNetwork->mutate(&game.networks[0]);
		newPastNetwork->mutate(&game.networks[1], 0.0, 0.0, 0.0, 0.0);
	}

	// the first game will always be the best network of the last round
	bestNetwork->mutate(&m_allGames[0].networks[0], 0.0, 0.0, 0.0, 0.0);
	newPastNetwork->mutate(&m_allGames[0].networks[1], 0.0, 0.0, 0.0, 0.0);
}


void Simulation::tickGames(bool& stop)
{
	if (!m_paused)
	{
		for (Game& game : m_allGames)
			stop = game.tick();
	}
}

void Simulation::endOfGenStats()
{
	++m_generationCount;

	if (m_generationCount == 1000)
	{
		std::cout << m_totalRunTime << "\n";
		//m_closeSim = true;
	}

	if (m_auto_save && m_generationCount % autoSaveFreq == 0)
	{
		saveNetworkData();
	}
}

void Simulation::resetGames()
{
	// getting the starting positions
	const std::vector<sf::Vector2f> positions = rearrangePositions(bounds, GameSettings::agentsPergame);
	

	for (Game& game : m_allGames)
	{
		game.initiliseGame(positions);
	}

	if (best_net_info.learnerPosition != sf::Vector2f{ 0.f, 0.f })
	{
		m_allGames[0].agents[0].position = best_net_info.learnerPosition;
		m_allGames[0].agents[1].position = best_net_info.trainerPosition;
	}

	for (Game& game : m_allGames) {for (Agent& agent : game.agents) {
		agent.gameStartPos = agent.position;
	}}
}


void Simulation::getTopNet()
{
	// sorting to get the best two networks
	best_net_info.score = 100000;
	best_net_info.Network = nullptr;

	for (unsigned i = 0; i < parrelelGames; i++)
	{
		Game& game = m_allGames[i];
		const float score = game.agents[0].network_score;

		if (score < best_net_info.score)
		{
			best_net_info.score = score;
			best_net_info.Network = &game.networks[0];
			best_net_info.learnerPosition = game.agents[0].gameStartPos;
			best_net_info.trainerPosition = game.agents[1].gameStartPos;

		}
	}
	if (m_generationCount % 10 == 0)
		std::cout << "best score for gen " << m_generationCount << ": " << best_net_info.score << "\n";

}

void Simulation::endFrame()
{
	// updating runtime statistics and ending the frame
	m_totalFrameCount++;
	m_totalRunTime += GetDelta();
}