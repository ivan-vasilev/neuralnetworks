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
 * Random Initializer for neural networks weights - all the connections between neurons are traversed and initialized
 */
public class NNRandomInitializer implements Serializable {

    private static final long serialVersionUID = 1L;

    protected RandomInitializer randomInitializer;

    /**
     * Random initializer for bias weights. If null, the default
     * randomInitializer will be used. if biasDefaultValue != null
     * biasDefaultValue has preference
     */
    protected RandomInitializer biasRandomInitializer;

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

    public NNRandomInitializer(RandomInitializer randomInitializer, RandomInitializer biasRandomInitializer) {
	super();
	this.randomInitializer = randomInitializer;
	this.biasRandomInitializer = biasRandomInitializer;
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
		if (Util.isBias(fc.getInputLayer())) {
		    if (biasDefaultValue != null) {
			fc.getConnectionGraph().forEach(i -> fc.getConnectionGraph().getElements()[i] = biasDefaultValue);
		    } else if (biasRandomInitializer != null) {
			biasRandomInitializer.initialize(fc.getConnectionGraph().getElements());
		    } else {
			randomInitializer.initialize(fc.getConnectionGraph().getElements());
		    }
		} else {
		    randomInitializer.initialize(fc.getConnectionGraph().getElements());
		}
	    } else if (cc.connection instanceof Conv2DConnection) {
		Conv2DConnection c = (Conv2DConnection) cc.connection;
		if (Util.isBias(c.getInputLayer())) {
		    if (biasDefaultValue != null) {
			c.getWeights().forEach(i -> c.getWeights().getElements()[i] = biasDefaultValue);
		    } else if (biasRandomInitializer != null) {
			biasRandomInitializer.initialize(c.getWeights().getElements());
		    } else {
			randomInitializer.initialize(c.getWeights().getElements());
		    }
		} else {
		    randomInitializer.initialize(c.getWeights().getElements());
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

    public RandomInitializer getBiasRandomInitializer() {
	return biasRandomInitializer;
    }

    public void setBiasRandomInitializer(RandomInitializer biasRandomInitializer) {
	this.biasRandomInitializer = biasRandomInitializer;
    }

    public Float getBiasDefaultValue() {
	return biasDefaultValue;
    }

    public void setBiasDefaultValue(Float biasDefaultValue) {
	this.biasDefaultValue = biasDefaultValue;
    }
}
