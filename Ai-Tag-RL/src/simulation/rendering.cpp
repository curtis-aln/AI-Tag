#include "Simulation.hpp"

#include <SFML/Graphics.hpp>
#include "../utility.hpp"


void Simulation::keyPressEvents(const sf::Keyboard::Key& event_key_code)
{
	const bool ctrl = sf::Keyboard::isKeyPressed(sf::Keyboard::LControl);

	switch (event_key_code)
	{
	case sf::Keyboard::Escape:
		m_closeSim = true;
		break;

	case sf::Keyboard::Space:
		m_paused = not m_paused;
		break;


	case sf::Keyboard::Key::D:
		m_debug = not m_debug;
		break;

	case sf::Keyboard::Key::U:
		m_allrender = not m_allrender;
		break;

	case sf::Keyboard::Key::A:
		m_auto_save = not m_auto_save;
		std::cout << "[Setting]: Autosave: " << m_auto_save << "\n";
		break;

	case sf::Keyboard::Key::V:
		m_debugValue = not m_debugValue;
		break;

	case sf::Keyboard::Key::F:
		fastForward = true;
		break;

	case sf::Keyboard::Key::S:
		if (ctrl)
			saveNetworkData();
		break;

	case sf::Keyboard::Key::L:
		if (ctrl)
			loadNetworkData();
		break;


	case sf::Keyboard::Key::R:
		m_rendering = not m_rendering;

		if (m_rendering == true)
			m_window.setFramerateLimit(100);
		else
			m_window.setFramerateLimit(999'999);
		break;

	default:
		break;

	}
}


void Simulation::pollEvents()
{
	sf::Event event{};
	while (m_window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
			m_closeSim = true;

		else if (event.type == sf::Event::KeyPressed)
			keyPressEvents(event.key.code);
	}
}


sf::Color Simulation::getColor(const bool tagged)
{
	return (tagged == true) ? AgentSettings::itColor : AgentSettings::notItColor;
}


void Simulation::renderAgents()
{
	for (unsigned i = 0; i < GameSettings::agentsPergame; i++)
	{
		Agent& agent = m_allGames[0].agents[i];
		m_agentRenderCircle.setPosition(agent.position - sf::Vector2f{AgentSettings::radius, AgentSettings::radius});
		m_agentRenderCircle.setFillColor(getColor(agent.tagged));
		m_agentRenderCircle.setOutlineColor(colors[i]);
		m_window.draw(m_agentRenderCircle);
	}
}


void Simulation::renderFrame()
{
	// Clearing the screen
	m_window.clear(windowColor);

	// drawing border
	m_window.draw(m_gameBorderRenderer);

	// Rendering Agents
	renderAgents();
	//m_agentRenderCircle.setFillColor(AgentSettings::itColor);
	//m_agentRenderCircle.setPosition(posToFollow::position - sf::Vector2f{AgentSettings::radius, AgentSettings::radius});
	//m_window.draw(m_agentRenderCircle);

	if (m_debug)
		debugAgents();


	m_window.display();
}


void Simulation::setWindowTitle()
{
	const sf::Int32 msPerFrame = m_clock.restart().asMilliseconds();
	if (msPerFrame != 0)
	{
		const sf::Uint16 fps = 1000 / msPerFrame;
		m_frameRateManager.updateFrameRates(fps);

		std::ostringstream oss;
		oss << simulationName << " " << fps << "fps, gen " << m_generationCount << ", time until next gen: " << m_allGames[0].timeRemaining  << " \n";
		const std::string stringFrameRate = oss.str();
		m_window.setTitle(stringFrameRate);
	}
}


void Simulation::debugAgents()
{
	for (Game& game : m_allGames)
	{
		for (Agent& agent : game.agents)
		{
			debugAgent(&agent);
		}

		if (!m_allrender)
			break;
	}
}


void Simulation::debugAgent(const Agent* agent)
{
	const sf::Vector2f position = agent->position;

	// Normalize the velocity vector
	const sf::Vector2f velocity = agent->m_velocity * 5.f;
	const sf::Vector2f accelaration = agent->m_accelaration * 1000.f;

	m_window.draw(makeLine(position, position + velocity, { 255, 0  , 0 }));
	m_window.draw(makeLine(position + velocity, position + velocity + accelaration, { 0, 255  , 0 }));


	sf::RenderStates states{};
	const auto value = static_cast<int>(roundToNearestN(agent->network_score, 1));
	scores.drawCenteredValue(agent->position, value, states);
}