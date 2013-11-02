package com.github.neuralnetworks.training;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.LayerCalculator;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;

public class BackPropagationTrainer extends OneStepTrainer<NeuralNetworkImpl> {

    private Map<Layer, Matrix> activations;

    public BackPropagationTrainer() {
	super();
	activations = new HashMap<Layer, Matrix>();
    }

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	activations = new HashMap<Layer, Matrix>();
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	NeuralNetwork nn = getNeuralNetwork();

//	for (Matrix m : activations.values()) {
//	    Util.fillArray(m.getElements(), 0);
//	}

	activations.put(nn.getInputLayer(), data.getInput());

	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	LayerCalculator lc = getLayerCalculator();
	lc.calculate(calculatedLayers, activations, nn.getOutputLayer());

	BackPropagation bp = getProperties().getParameter(Constants.BACKPROPAGATION);
	Matrix outputErrorDerivative = bp.getOutputErrorDerivative(activations.get(nn.getOutputLayer()), data.getTarget());
	bp.backPropagate(activations, outputErrorDerivative, nn);
    }
}
