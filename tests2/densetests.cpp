#include <gtest/gtest.h>
#include "neurallayers.hpp"
#include <stdexcept>

TEST(DenseTests, buildDenseLayer) {
    const size_t layerSize = 4;
    const size_t depth = 2;
    Dense dense(layerSize, depth);

    // Check variables are set
    EXPECT_EQ(dense.layerSize, layerSize);
    EXPECT_EQ(dense.depth, depth);

    EXPECT_EQ(dense.valueSize, layerSize*depth);
    EXPECT_EQ(dense.weightSize, layerSize*layerSize*depth);
    EXPECT_EQ(dense.biasSize, layerSize*depth);

    // Check values are generated
    try {
        // Values should just be set to 0
        float valueSum = 0;
        for (size_t valueIndex = 0; valueIndex < dense.valueSize; valueIndex++)
            valueSum += dense.values.at(valueIndex);
        EXPECT_EQ(valueSum, 0);
        
        // Weights should be random, some can be 0, so we test if all the values summed aren't 0
        float weightSum = 0;
        for (size_t weightIndex = 0; weightIndex < dense.weightSize; weightIndex++)
            weightSum += dense.weights.at(weightIndex) * dense.weights.at(weightIndex); // Squared to keep the value positive
        EXPECT_TRUE(weightSum != 0);

        // Bias should be random, so we also test for not 0
        float biasSum = 0;
        for (size_t biasIndex = 0; biasIndex < dense.biasSize; biasIndex++)
            biasSum += dense.bias.at(biasIndex) * dense.bias.at(biasIndex); // Squared to keep the value positive
        EXPECT_TRUE(biasSum != 0);
    } catch (const std::out_of_range& error) {
        ADD_FAILURE() << "Unexpected out-of-range access: " << error.what();
    }
}

TEST(DenseTests, invalidBuild) {
    EXPECT_THROW(Dense(0, 2), std::invalid_argument);
    EXPECT_THROW(Dense(4, 0), std::invalid_argument);
}

TEST(DenseTests, invalidActivationInput) {
    const size_t layerSize = 4;
    const size_t depth = 2;
    Dense dense(layerSize, depth);

    const size_t inputSize = layerSize+1;
    vector<float> inputs(inputSize, 1);
    EXPECT_THROW(dense.activate(inputs.data(), inputSize), std::invalid_argument);
}

TEST(DenseTests, activation) {
    const size_t layerSize = 4;
    const size_t depth = 2;
    Dense dense(layerSize, depth);

    const size_t inputSize = layerSize;
    vector<float> inputs(inputSize, 1);
    dense.activate(inputs.data(), inputSize);
}