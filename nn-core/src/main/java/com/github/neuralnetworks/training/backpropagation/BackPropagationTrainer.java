package com.github.neuralnetworks.training.backpropagation;

import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.TensorFactory;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Base backpropagation one step trainer
 * It has two additional parameters:
 * BackPropagationLayerCalculator for the backpropagation phase
 * OutputErrorDerivative for calculating the derivative of the output error
 * This allows for various implementations of these calculators to be used (for example via GPU or other)
 */
public class BackPropagationTrainer<N extends NeuralNetwork> extends OneStepTrainer<N> {

    private static final long serialVersionUID = 1L;

    protected ValuesProvider activations;
    protected ValuesProvider backpropagation;
    protected TrainingInputData input;

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	activations = TensorFactory.tensorProvider(getNeuralNetwork(), getTrainingBatchSize(), Environment.getInstance().getUseSharedMemory());
	activations.add(getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE), activations.get(getNeuralNetwork().getOutputLayer()).getDimensions());
	backpropagation = TensorFactory.tensorProvider(getNeuralNetwork(), getTrainingBatchSize(), Environment.getInstance().getUseSharedMemory());
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.training.OneStepTrainer#learnInput(com.github.neuralnetworks.training.TrainingInputData)
     * The training example is propagated forward through the network (via the LayerCalculator lc) and the results are stored.
     * After that the error is backpropagated (via BackPropagationLayerCalculator blc).
     */
    @Override
    protected void learnInput(int batch) {
	// forward
	NeuralNetwork nn = getNeuralNetwork();
	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, activations);

	// backward
	OutputErrorDerivative d = getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE);
	d.getOutputErrorDerivative(activations.get(nn.getOutputLayer()), activations.get(d), backpropagation.get(nn.getOutputLayer()));
	calculatedLayers.clear();
	calculatedLayers.add(nn.getOutputLayer());
	BackPropagationLayerCalculator blc = getBPLayerCalculator();
	blc.backpropagate(nn, calculatedLayers, activations, backpropagation);
    }

    @Override
    protected TrainingInputData getInput() {
	if (input == null) {
	    input = new TrainingInputDataImpl(activations.get(getNeuralNetwork().getInputLayer()), activations.get(getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE)));
	}

	return input;
    }

    public BackPropagationLayerCalculator getBPLayerCalculator() {
	return getProperties().getParameter(Constants.BACKPROPAGATION);
    }

    public void setBPLayerCalculator(BackPropagationLayerCalculator bplc) {
	getProperties().setParameter(Constants.BACKPROPAGATION, bplc);
    }
}
