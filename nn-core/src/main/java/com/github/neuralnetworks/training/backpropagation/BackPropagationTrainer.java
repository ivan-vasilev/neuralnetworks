package com.github.neuralnetworks.training.backpropagation;

import java.util.Set;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.ConnectionCalculatorFullyConnected;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.events.TrainingFinishedEvent;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.Util;

/**
 * Base backpropagation one step trainer
 * It has two additional parameters:
 * BackPropagationLayerCalculator for the backpropagation phase
 * OutputErrorDerivative for calculating the derivative of the output error
 * This allows for various implementations of these calculators to be used (for example via GPU or other)
 */
public class BackPropagationTrainer<N extends NeuralNetwork> extends OneStepTrainer<N> implements TrainingEventListener {

    private static final long serialVersionUID = 1L;

    protected ValuesProvider activations;
    protected ValuesProvider backpropagation;
    protected TrainingInputData input;

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	NeuralNetwork nn = getNeuralNetwork();
	activations = TensorFactory.tensorProvider(nn, getTrainingBatchSize(), Environment.getInstance().getUseDataSharedMemory());
	activations.add(getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE), activations.get(getNeuralNetwork().getOutputLayer()).getDimensions());
	backpropagation = TensorFactory.tensorProvider(nn, getTrainingBatchSize(), Environment.getInstance().getUseDataSharedMemory());

	float dropoutRate = properties.getParameter(Constants.DROPOUT_RATE);

	if (dropoutRate > 0) {
	    LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
	    nn.getConnections().stream().filter(c -> c instanceof FullyConnected && c.getInputLayer() != nn.getInputLayer() && !Util.isBias(c.getInputLayer())).forEach(c -> {
		ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(c.getOutputLayer());
		cc.setDropoutRate(dropoutRate);
	    });

	    addEventListener(this);
	}
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

    @Override
    public void handleEvent(TrainingEvent event) {
	if (event instanceof TrainingFinishedEvent) {
	    float dropoutRate = properties.getParameter(Constants.DROPOUT_RATE);

	    if (dropoutRate > 0) {
		NeuralNetwork nn = getNeuralNetwork();

		LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
		nn.getConnections().stream().filter(c -> c instanceof FullyConnected && c.getInputLayer() != nn.getInputLayer() && !Util.isBias(c.getInputLayer())).forEach(c -> {
		    ConnectionCalculatorFullyConnected cc = (ConnectionCalculatorFullyConnected) lc.getConnectionCalculator(c.getOutputLayer());
		    cc.setDropoutRate(0);
		    FullyConnected fc = (FullyConnected) c;
		    fc.getWeights().forEach(i -> fc.getWeights().getElements()[i] = fc.getWeights().getElements()[i] * (1 - dropoutRate));
		});
	    }
	}
    }

    public BackPropagationLayerCalculator getBPLayerCalculator() {
	return getProperties().getParameter(Constants.BACKPROPAGATION);
    }

    public void setBPLayerCalculator(BackPropagationLayerCalculator bplc) {
	getProperties().setParameter(Constants.BACKPROPAGATION, bplc);
    }
}
