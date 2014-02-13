package com.github.neuralnetworks.training.random;

import java.io.Serializable;
import java.util.List;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.GraphConnections;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.util.Util;

/**
 * Random Initializer for neural networks
 */
public class NNRandomInitializer implements Serializable {

    private static final long serialVersionUID = 1L;

    protected RandomInitializer randomInitializer;

    /**
     * If this is != null all bias weights are initialized with this value
     */
    protected Float biasDefaultValue;

    public NNRandomInitializer() {
	super();
    }

    public NNRandomInitializer(RandomInitializer randomInitializer) {
	super();
	this.randomInitializer = randomInitializer;
    }

    public NNRandomInitializer(RandomInitializer randomInitializer, Float biasDefaultValue) {
	super();
	this.randomInitializer = randomInitializer;
	this.biasDefaultValue = biasDefaultValue;
    }

    public void initialize(NeuralNetwork nn) {
	List<ConnectionCandidate> ccs = new BreadthFirstOrderStrategy(nn, nn.getInputLayer()).order();
	for (ConnectionCandidate cc : ccs) {
	    if (cc.connection instanceof GraphConnections) {
		GraphConnections fc = (GraphConnections) cc.connection;
		if (biasDefaultValue != null && Util.isBias(fc.getInputLayer())) {
		    Util.fillArray(fc.getConnectionGraph().getElements(), biasDefaultValue);
		} else {
		    randomInitializer.initialize(fc.getConnectionGraph().getElements());
		}
	    } else if (cc.connection instanceof Conv2DConnection) {
		Conv2DConnection c = (Conv2DConnection) cc.connection;
		if (biasDefaultValue != null && Util.isBias(c.getInputLayer())) {
		    Util.fillArray(c.getWeights(), biasDefaultValue);
		} else {
		    randomInitializer.initialize(c.getWeights());
		}
		
	    }
	}
    }

    public RandomInitializer getRandomInitializer() {
	return randomInitializer;
    }

    public void setRandomInitializer(RandomInitializer randomInitializer) {
	this.randomInitializer = randomInitializer;
    }

    public Float getBiasDefaultValue() {
	return biasDefaultValue;
    }

    public void setBiasDefaultValue(Float biasDefaultValue) {
	this.biasDefaultValue = biasDefaultValue;
    }
}
