package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.types.ConnectionFactory;
import com.github.neuralnetworks.architecture.types.DefaultNeuralNetwork;

/**
 * Tests for convolutional networks
 */
public class CNNTest {

    @Test
    public void testDimensions() {
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(4, 4, 3), 2, 2, 2);

	ConvGridLayer output = (ConvGridLayer) c.getOutputLayer();

	assertEquals(3, output.getFeatureMapColumns(), 0);
	assertEquals(3, output.getFeatureMapRows(), 0);
	assertEquals(2, output.getFilters(), 0);
    }

    @Test
    public void testConvolutions() {
	DefaultNeuralNetwork nn = new DefaultNeuralNetwork();
	Conv2DConnection c = ConnectionFactory.convConnection(new ConvGridLayer(4, 4, 3), 2, 2, 2);
	nn.addConnection(c);
    }
}
