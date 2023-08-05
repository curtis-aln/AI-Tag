#pragma once
#include <nlohmann/json.hpp>

#include "utility.hpp"
#include "settings.hpp"

#pragma once

#include <vector>


struct Layer
{
	Layer(const uint64_t neurons_count, const uint64_t prev_count)
		: weights(neurons_count)
		, values(neurons_count)
		, bias(neurons_count)
	{
		for (std::vector<float>& v : weights) {
			v.resize(prev_count);
		}
	}

	uint64_t getNeuronsCount() const
	{
		return bias.size();
	}

	uint64_t getWeightsCount() const
	{
		return weights.front().size();
	}

	void process(const std::vector<float>& inputs)
	{
		const uint64_t neurons_count = bias.size();
		const uint64_t inputs_count = inputs.size();
		// For each neuron
		for (uint64_t i(0); i < neurons_count; ++i) {
			float result = bias[i];
			// Compute weighted sum of inputs
			for (uint64_t j(0); j < inputs_count; ++j) {
				result += weights[i][j] * inputs[j];
			}
			// Output result
			values[i] = tanh(4.0f * result);
		}
	}

	void print() const
	{
		std::cout << "--- layer ---" << std::endl;
		const uint64_t neurons_count = values.size();
		for (uint64_t i(0); i < neurons_count; ++i) {
			const uint64_t inputs_count = getWeightsCount();
			// Compute weighted sum of inputs
			std::cout << "Neuron " << i << " bias " << bias[i] << std::endl;
			for (uint64_t j(0); j < inputs_count; ++j) {
				std::cout << weights[i][j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "--- end ---\n" << std::endl;
	}

	std::vector<std::vector<float>> weights;
	std::vector<float> values;
	std::vector<float> bias;
};


struct Network
{
	Network()
		: input_size(0)
		, last_input(0)
	{}

	Network(const uint64_t input_size_)
		: input_size(input_size_)
		, last_input(input_size_)
	{}

	Network(const std::vector<uint64_t>& layers_sizes)
		: input_size(layers_sizes[0])
		, last_input(input_size)
	{
		for (uint64_t i(1); i < layers_sizes.size(); ++i) {
			addLayer(layers_sizes[i]);
		}
	}

	void addLayer(const uint64_t neurons_count)
	{
		if (!layers.empty()) {
			layers.emplace_back(neurons_count, layers.back().getNeuronsCount());
		}
		else {
			layers.emplace_back(neurons_count, input_size);
		}
	}

	const std::vector<float>& execute(const std::vector<float>& input)
	{
		last_input = input;
		if (input.size() == input_size) {
			layers.front().process(input);
			const uint64_t layers_count = layers.size();
			for (uint64_t i(1); i < layers_count; ++i) {
				layers[i].process(layers[i - 1].values);
			}
		}

		return layers.back().values;
	}

	uint64_t getParametersCount() const
	{
		uint64_t result = 0;
		for (const Layer& layer : layers) {
			result += layer.bias.size() * (1 + layer.weights.front().size());
		}
		return result;
	}

	static uint64_t getParametersCount(const std::vector<uint64_t>& layers_sizes)
	{
		uint64_t count = 0;
		for (uint64_t i(1); i < layers_sizes.size(); ++i) {
			count += layers_sizes[i] * (1 + layers_sizes[i - 1]);
		}
		return count;
	}

	uint64_t input_size;
	std::vector<Layer> layers;
	std::vector<float> last_input;
};



using Weight = float;
using Bias   = float;

using Node  = Weight[NetSettings::largestLayer];
using LayerWeights = Node[NetSettings::largestnonInpLayer];
using LayerBiases = Bias[NetSettings::largestnonInpLayer];


class Neural9Network : NetSettings
{
    static void mutate_value(float& value, const float rate, const float range)
    {
        if (RandomDist::rand01float() < rate)
        {
            value += RandomDist::randRange(-range, range);
        }
    }


public:
    Neural9Network()
    {
        // initilising weights
    	mutate(this, 0.4f, 0.4f, 0.4f, 0.4f);
    }

    void compute_output()
    {
    	outputs = inputs;

        for (unsigned layer_idx = 0; layer_idx < NetworkLayers - 1; ++layer_idx) // each network layer
        {
	        for (unsigned node_idx = 0; node_idx < NN_dims[layer_idx + 1]; ++node_idx) // each node in the network
	        {
                float dotted = biases[layer_idx][node_idx];

                for (unsigned weight_idx = 0; weight_idx < NN_dims[layer_idx]; ++weight_idx) // calculating the dot product
                    dotted += weights[layer_idx][node_idx][weight_idx] * outputs[weight_idx];

                //temp[node_idx] = dotted;
                temp[node_idx] = tanh(dotted * 2.0);
                //temp[node] = (0 < layer && layer < NetworkLayers - 1) ? std::min(0.0, dotted) : dotted;
	        }
        	outputs = temp;
        }
    }


    void mutate(Neural9Network* net, 
        const float w_rate = weight_mutation_rate, const float w_range = weight_mutation_range,
        const float b_rate = bias_mutation_rate, const float b_range = bias_mutation_range) const
    {
        // `NetworkLayers - 1` as we do not have input weights
        for (unsigned layer = 0; layer < NetworkLayers - 1; ++layer) // each network layer
        {
            for (unsigned node = 0; node < NN_dims[layer + 1]; ++node)
            {
                for (unsigned weight = 0; weight < NN_dims[layer]; ++weight)
                {
                    net->weights[layer][node][weight] = weights[layer][node][weight];
                    mutate_value(net->weights[layer][node][weight], w_rate, w_range);
                }
            }

            for (unsigned bias = 0; bias < NN_dims[layer + 1]; ++bias)
            {
                net->biases[layer][bias] = biases[layer][bias];
                mutate_value(net->biases[layer][bias], b_rate, b_range);
            }
        }
    }

    void jsonFormat(nlohmann::json& writeTo)
    {
        return writeTo.push_back({ {"weights", weights}, {"biases", biases}});
    }


public:
    LayerWeights weights[NetworkLayers - 1];
    LayerBiases biases[NetworkLayers-1];

    std::array<float, largestLayer> inputs  = {};
    std::array<float, largestLayer> outputs = {};
    std::array<float, largestLayer> temp    = {};
};


class ReinforcementLearning
{
public:
    static constexpr int8_t   snapshot_window = 10;           // amount of networks stored in history, oldest gets overwrittten
    static constexpr uint16_t snapshot_frequency = 250;       // how often a network gets logged, stores a play strategy
    static constexpr uint16_t swap_steps = 5;                 // how often the network the learning agent plays against changes
    static constexpr float    play_lastest_model_ratio = 0.5; // chance of choosing a random past network over the latest one
    std::vector<Neural9Network> policy{};                      // where all the past networks are stored

    uint8_t current_index = 0; // incharge of overwriting older networks
    uint8_t actual_size = 0;
    Neural9Network* chosenNetwork = nullptr;


    ReinforcementLearning()
    {
        policy.resize(snapshot_window);

        // the first neural network in the policy should be completly random
        add_neural_network({}, snapshot_frequency);
    }


    void add_neural_network(const Neural9Network& neural_network, const unsigned genaration_count)
    {
        if (genaration_count % snapshot_frequency != 0)
            return;

        std::cout << "[Notice]: Adding Neural Network \n";
        policy[current_index] = neural_network;
        increment();
    }


    Neural9Network* get_network(const unsigned genaration_count)
    {
        if (chosenNetwork == nullptr || genaration_count % swap_steps == 0)
        {
            if (RandomDist::rand01float() > play_lastest_model_ratio && actual_size > 1)
                chosenNetwork = &policy[RandomDist::randRange(0u, static_cast<unsigned>(actual_size))];
            chosenNetwork = &policy[current_index];
        }
        return chosenNetwork;
    }


    void reset_information() { actual_size = 0; current_index = 0; chosenNetwork = nullptr; }
    void increment()
    {
        actual_size += actual_size < policy.size();
        current_index += (current_index == snapshot_window - 1) ? -(snapshot_window - 1) : 1;
    }
};