#pragma once

#include <nlohmann/json.hpp>
#include <fstream> // for std::ofstream
#include <cmath>
#include <boost/functional/hash.hpp>
#include <iostream>
#include <random>


// a class used to get a more stable and accurate reading of framerates by averaging out the last N
// frame rates and returning that
template<const unsigned resolution>
struct BetterFrameRates
{
	sf::Uint16 framerates[resolution] = {};
	sf::Uint8 size = 0;    // tracks the current "size" of the array
	sf::Uint8 counter = 0; // manages frame rate overwriting

	int getFrameRate()
	{
		if (size == 0) return 0;

		unsigned sum = 0;
		for (const unsigned val : framerates) { sum += val; }
		return sum / size;
	}

	void updateFrameRates(const sf::Uint16 frameRate)
	{
		if (counter >= resolution)
			counter = 0;

		if (size < resolution) ++size;
		framerates[counter++] = frameRate;
	}
};


// a simple class used for managing time frames, the GetDelta function updates the time
// and returns the change in time since the last call, so call tagged at set frequent intervals
struct DeltaTime
{
	DeltaTime()
	{
		m_start = std::chrono::high_resolution_clock::now();
	}

	double GetDelta()
	{
		const auto currentTime = std::chrono::high_resolution_clock::now();
		const auto delta = currentTime - m_start;
		m_start = currentTime;
		return std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
	}

private:
	std::chrono::high_resolution_clock::time_point m_start;
};


// the dot product of two vectors (2d)
template<typename type>
inline type dot(const sf::Vector2<type>& v1, const sf::Vector2<type>& v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}

// the dot product of two vectors (containers)
template<typename type>
inline type dot(const std::vector<type>& v1, const std::vector<type>& v2)
{
	type result = 0.0;

	for (unsigned i = 0; i < v1.size(); i++)
		result += v1[i] * v2[i];

	return result;
}

// adds "vecToAdd" to the vector "vector"
template<typename type>
inline void addToVector(std::vector<type>& vector, const std::vector<type>& vecToAdd)
{
	for (unsigned i = 0; i < vector.size(); i++)
	{
		vector[i] += vecToAdd[i];
	}
}

inline double sigmoid(const double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double scaledSigmoid(const double x) { return 2.0 * sigmoid(x) - 1.0; }

// A dot Product made for neural networks, tagged dots a 2d vector with a 1d vector
template<typename type>
inline std::vector<type>& dotNetwork(
	std::vector<type>& writeTo, 
	const std::vector<type>& inputs, 
	const std::vector<std::vector<type>>& weights,
	const std::vector<type>& biases)
{
	unsigned i = 0;
	for (const auto& weight : weights)
	{
		writeTo[i] = dot(weight, inputs) + biases[i];
		++i;
	}
	return writeTo;
}

// length of a vector
inline float length(const sf::Vector2f& v)
{
	return std::sqrt(dot(v, v));
}

inline float lengthSquared(const sf::Vector2f& vec)
{
	return vec.x * vec.x + vec.y * vec.y;
}

// normilises the length of a vector to a specified size, note the std::sqrt here
template<typename type>
inline sf::Vector2<type> normalise(const sf::Vector2<type>& vector, const type size = 1)
{
	sf::Vector2<type> normalised = vector;

	// Normalize the velocity vector
	if (const float length = std::sqrt(vector.x * vector.x + vector.y * vector.y); length > 0)
		normalised /= length;

	normalised *= size;
	return normalised;
}



inline sf::Vector2u clipToGrid(const sf::Vector2u& position, const sf::Vector2u& tileSize)
{
	const sf::Vector2u index(position.x / tileSize.x, position.y / tileSize.y);
	return { index.x * tileSize.x, index.y * tileSize.y};
}


inline float distSquared(const sf::Vector2f& positionA, const sf::Vector2f& positionB)
{
	const sf::Vector2f delta = positionB - positionA;
	return delta.x * delta.x + delta.y * delta.y;
}

template<typename type>
inline sf::Rect<type> resizeRect(const sf::Rect<type>& rect, const sf::Vector2<type>& resize)
{
	return {
		rect.left + resize.x,
		rect.top + resize.y,
		rect.width - resize.x * 2.f,
		rect.height - resize.y * 2.f
	};
}

inline sf::Vector2f getMousePosAsFloat(const sf::RenderWindow& window)
{
	const sf::Vector2f mousePosition = {
		static_cast<float>(sf::Mouse::getPosition(window).x),
		static_cast<float>(sf::Mouse::getPosition(window).y)
	};
	return mousePosition;
}


inline void displayFrameRate(sf::RenderWindow& window, const std::string& title, const unsigned fps)
{
	std::ostringstream oss;
	oss << title << " " << fps << "fps \n";
	const std::string stringFrameRate = oss.str();
	window.setTitle(stringFrameRate);
}

template<typename type>
inline type roundToNearestN(const type value, const unsigned decimal_places) {
	const type multiplier = pow(static_cast<type>(10), decimal_places);
	return round(value * multiplier) / multiplier;
}

inline sf::VertexArray makeLine(const sf::Vector2f& point1, const sf::Vector2f& point2, const sf::Color color)
{
	sf::VertexArray line(sf::Lines, 2);
	line[0].position = point1;
	line[1].position = point2;
	line[0].color = color;
	line[1].color = color;
	return line;
}

inline std::string formatVariables(const std::vector<std::pair<std::string, double>>& variables) {
	std::ostringstream oss;
	oss.precision(2);
	oss << std::fixed;
	for (const auto& [fst, snd] : variables) {
		oss << fst << ": " << snd << ", ";
	}
	const std::string result = oss.str();
	// Remove the last comma and space
	return result.substr(0, result.size() - 2);
}


inline void drawRectOutline(sf::Rect<float>& rect, sf::RenderWindow& window, const sf::RenderStates& renderStates)
{
	sf::VertexArray lines(sf::Lines, 8);

	// Top line
	lines[0].position = { rect.left, rect.top };
	lines[1].position = { rect.left + rect.width, rect.top };

	// Right line
	lines[2].position = { rect.left + rect.width, rect.top };
	lines[3].position = { rect.left + rect.width, rect.top + rect.height };

	// Bottom line
	lines[4].position = { rect.left + rect.width, rect.top + rect.height };
	lines[5].position = { rect.left, rect.top + rect.height };

	// Left line
	lines[6].position = { rect.left, rect.top + rect.height };
	lines[7].position = { rect.left, rect.top };

	window.draw(lines, renderStates);
}



inline nlohmann::json loadJsonData(const std::string& fileReadWriteName)
{
	std::ifstream ifs(fileReadWriteName);

	if (!ifs.is_open()) {
		std::cerr << "Failed to open the file." << std::endl;
		return nlohmann::json{};
	}

	// Read the content of the JSON file into a JSON object
	nlohmann::json jsonData;
	ifs >> jsonData;

	ifs.close();

	return jsonData;
}


inline sf::Vector2f jsonToVector(const nlohmann::json& jsonPosition)
{
	return {jsonPosition["x"], jsonPosition["y"]};
}

inline nlohmann::json vectorToJson(const sf::Vector2f& vector)
{
	return { {"x", vector.x}, {"y", vector.y} };
}

inline sf::Color jsonToColor(const nlohmann::json& jsonColor)
{
	return { jsonColor["r"], jsonColor["g"], jsonColor["b"]};
}

inline nlohmann::json rectToJson(const sf::Rect<float>& rect)
{
	return { {"x", rect.left}, {"y", rect.top}, {"w", rect.width}, {"h", rect.height} };
}

inline sf::Rect<float> jsonToRect(const nlohmann::json& json)
{
	return {json["x"], json["y"], json["w"], json["h"]};
}

inline nlohmann::json colorToJson(const sf::Color color)
{
	return { 
	{"r", color.r },
	{"g", color.g },
	{"b", color.b }};
}

inline std::vector<float> jsonToVectorCont(const nlohmann::json& jsonData)
{
	std::vector<float> newData;
	std::cout << jsonData << "\n";
	std::cout << jsonData.size() << "\n";

	for (const auto& i : jsonData)
		newData.push_back(i);
	
	return newData;
}


inline float generaeteUniqueIdentifier(const std::vector<float>& numbers)
{
	std::size_t seed = 0;
	boost::hash_range(seed, numbers.begin(), numbers.end());
	return static_cast<float>(seed) / static_cast<float>(std::numeric_limits<std::size_t>::max());
}


inline float cosineSimilarity(const std::vector<float>& weights1, const std::vector<float>& weights2)
{
	if (weights1.size() != weights2.size())
		return 0.0f;

	float dotProduct = 0.0f;
	float normWeights1 = 0.0f;
	float normWeights2 = 0.0f;

	for (std::size_t i = 0; i < weights1.size(); ++i)
	{
		dotProduct += weights1[i] * weights2[i];
		normWeights1 += weights1[i] * weights1[i];
		normWeights2 += weights2[i] * weights2[i];
	}

	normWeights1 = std::sqrt(normWeights1);
	normWeights2 = std::sqrt(normWeights2);

	if (normWeights1 == 0.0f || normWeights2 == 0.0f)
		return 0.0f;

	return dotProduct / (normWeights1 * normWeights2);
}

inline float generateUniqueIdentifier(const std::vector<float>& weights)
{
	// Calculate the similarity between weights and a reference vector (e.g., average weights)
	const std::vector referenceWeights(weights.size(), 0.5f);  // Replace with your reference weights
	const float similarity = cosineSimilarity(weights, referenceWeights);

	// Map the similarity value to a unique value between 0 and 1
	return (similarity + 1.0f) / 2.0f;
}


template <class E, unsigned max>
struct container_vector
{
	E* array[max] = {};
	uint8_t size = 0;

	void add(E* value)
	{
		if (size >= max) return;
		array[size++] = value;
	}

	[[nodiscard]] E* at(const unsigned index) const { return array[index]; }
};


struct CircularBorder
{
	sf::Vector2f position;
	float radius;


	[[nodiscard]] sf::Rect<float> asRect() const
	{
		return {
			position.x - radius,
			position.y - radius,
			position.x + radius,
			position.y + radius };
	}

	[[nodiscard]] bool contains(const sf::Vector2f& otherPosition) const
	{
		return distSquared(position, otherPosition) < radius * radius;
	}
};




// mapping the coordinate to have the circle center be (0, 0) and scaling tagged down
inline sf::Vector2f relativePosToCircle(const CircularBorder& circle, const sf::Vector2f& position)
{
	return (position - circle.position) / circle.radius;
}


inline void accurate_speed_limit(sf::Vector2f& vec, const float maxSpeed)
{
	const float speedSq = vec.x * vec.x + vec.y * vec.y;

	if (speedSq > maxSpeed * maxSpeed)
	{
		const float speed = std::sqrt(speedSq);
		vec *= maxSpeed / speed;
	}
}

template<typename type, unsigned size>
inline type getLargestValue(std::array<type, size>* array)
{
	type largest = 0;
	for (type value : array)
	{
		if (value > largest)
			largest = value;
	}
	return largest;
}

template <typename  type, size_t N>
void addArray(const std::array<type, N>& array1, const std::array<type, N>& array2)
{
	for (size_t i = 0; i < N; ++i) 
		array1[i] += array2[i];
}


template<unsigned size1, unsigned size2>
inline double dot(const std::array<double, size1>& v1, const std::array<double, size2>& v2, const unsigned size)
{
	double result = 0.0;

	for (unsigned i = 0; i < size; i++)
		result += v1[i] * v2[i];

	return result;
}

inline bool speed_limit(sf::Vector2f& vec, const float maxSpeed)
{
	bool capped = false;
	if (vec.x > maxSpeed) { vec.x = maxSpeed; capped = true; }
	else if (vec.x < -maxSpeed) { vec.x = -maxSpeed; capped = true; }

	if (vec.y > maxSpeed) { vec.y = maxSpeed; capped = true; }
	else if (vec.y < -maxSpeed) { vec.y = -maxSpeed; capped = true; }
	return capped;
}


inline bool border(const CircularBorder& border, sf::Vector2f& Position, const float radius)
{
	const float rad = border.radius - radius;
	const float distToCenter = distSquared(border.position, Position);
	const float radiusSQ = rad * rad;

	if (distToCenter > radiusSQ)
	{
		sf::Vector2f centerToObj = Position - border.position;
		centerToObj = normalise(centerToObj);
		Position = border.position + centerToObj * rad;
		return true;
	}
	return false;
}

class FontManager
{
	sf::Font m_font{};
	sf::Text text{};
	sf::RenderWindow* m_window;

public:
	FontManager(sf::RenderWindow* window, const unsigned font_size) : m_window(window)
	{
		if (!m_font.loadFromFile("Calibri.ttf"))
		{
			std::cout << "[ERROR]: failed to load font \n";
			return;
		}

		text.setCharacterSize(font_size);
		text.setFont(m_font);
		text.setFillColor(sf::Color::White);
	}

	void drawCenteredValue(const sf::Vector2f& position, const int value, sf::RenderStates& states)
	{
		text.setString(std::to_string(value));

		// Calculate the position for centering the text
		const sf::FloatRect textBounds = text.getLocalBounds();
		const sf::Vector2f textPosition(position.x - textBounds.width / 2, position.y - textBounds.height / 2);
		text.setPosition(textPosition);

		m_window->draw(text, states);
	}
};



// a wrapper to make generating random floats and integers more convinient
inline static std::random_device dev;
inline static std::mt19937 rng{ dev() }; // random number generator
struct RandomDist
{
	// random engines
	inline static std::uniform_real_distribution<float> float01_dist{ 0.f, 1.f };
	inline static std::uniform_real_distribution<float> float11_dist{ -1.f, 1.f };
	inline static std::uniform_int<int> int01_dist{ 0, 1 };
	inline static std::uniform_int<int> int11_dist{ -1, 1 };

	// basic random functions 11 = range(-1, 1), 01 = range(0, 1)
	static float rand11float() { return float11_dist(rng); }
	static float rand01float() { return float01_dist(rng); }
	static int   rand01int() { return int01_dist(rng); }
	static int   rand11int() { return int11_dist(rng); }


	// more complex random generation. specified ranges

	template <typename Type>
	static Type randRange(const Type min, const Type max)
	{
		// Check if the Type is an integer type
		if constexpr (std::is_integral_v<Type>)
		{
			std::uniform_int_distribution<Type> int_dist{ min, max };
			return int_dist(rng);
		}
		else
		{
			std::uniform_real_distribution<Type> float_dist{ min, max };
			return float_dist(rng);
		}
	}

	// random SFML::Vector<Type>
	static sf::Color randColor(const sf::Vector3<int> rgb_min = { 0, 0, 0 },
		const sf::Vector3<int> rgb_max = { 255 ,255, 255 })
	{
		return {
			static_cast<sf::Uint8>(randRange(rgb_min.x, rgb_max.x)), // red value
			static_cast<sf::Uint8>(randRange(rgb_min.y, rgb_max.y)), // green value
			static_cast<sf::Uint8>(randRange(rgb_min.x, rgb_max.z))  // blue value
		};
	}

	template<typename Type> // random SFML::Vector<Type>
	static sf::Vector2<Type> randVector(const Type min, const Type max)
	{
		return { randRange(min, max), randRange(min, max) };
	}

	template<typename Type> // random position inside of a rect
	static sf::Vector2<Type> randPosInRect(const sf::Rect<Type>& rect)
	{
		return { randRange(rect.left, rect.left + rect.width),
				 randRange(rect.top, rect.top + rect.height) };
	}
};



inline sf::Vector2f randPointInCircle(const CircularBorder& border)
{
	sf::Vector2f position = { 0, 0 };
	while (true)
	{
		position = RandomDist::randPosInRect(border.asRect());
		if (border.contains(position)) break;
	}
	return position;
}

inline sf::Vector2f randPointOutCircle(const CircularBorder& border)
{
	sf::Vector2f position = { 0, 0 };
	while (true)
	{
		position = RandomDist::randPosInRect(border.asRect());
		if (!border.contains(position)) break;
	}
	return position;
}


// Function to rearrange positions of positions along the circumference of a circle
inline std::vector<sf::Vector2f> rearrangePositions(const CircularBorder& circle, const unsigned position_count)
{
	const float angleStep = 360.0f / static_cast<float>(position_count);
	float angle = RandomDist::randRange(0.f, 360.f);

	std::vector<sf::Vector2f> positions(position_count);
	for (unsigned i = 0; i < position_count; ++i)
	{
		constexpr float pi = 3.14159f;
		const float posX = circle.position.x + circle.radius * std::cos(angle * pi / 180.0f);
		const float posY = circle.position.y + circle.radius * std::sin(angle * pi / 180.0f);
		positions[i] = sf::Vector2f(posX, posY);

		angle += angleStep;
	}

	return positions;
}