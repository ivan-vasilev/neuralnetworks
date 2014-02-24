package com.github.neuralnetworks.training.backpropagation;

import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Base backpropagation one step trainer
 * It has two additional parameters:
 * BackPropagationLayerCalculator for the backpropagation phase
 * OutputErrorDerivative for calculating the derivative of the output error
 * This allows for various implementations of these calculators to be used (for example via GPU or other)
 */
public class BackPropagationTrainer<N extends NeuralNetwork> extends OneStepTrainer<N> {

    private ValuesProvider activations;
    private ValuesProvider backpropagation;

    public BackPropagationTrainer() {
	super();
	activations = new ValuesProvider();
	backpropagation = new ValuesProvider();
    }

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	activations = new ValuesProvider();
	backpropagation = new ValuesProvider();
    }

    /* (non-Javadoc)
     * @see com.github.neuralnetworks.training.OneStepTrainer#learnInput(com.github.neuralnetworks.training.TrainingInputData)
     * The training example is propagated forward through the network (via the LayerCalculator lc) and the results are stored.
     * After that the error is backpropagated (via BackPropagationLayerCalculator blc).
     */
    @Override
    protected void learnInput(TrainingInputData data, int batch) {
	propagateForward(data.getInput());
	propagateBackward(data.getTarget());
    }

    public void propagateForward(Matrix input) {
	NeuralNetwork nn = getNeuralNetwork();
	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	activations.addValues(nn.getInputLayer(), input);
	nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, activations);
    }

    public void propagateBackward(Matrix target) {
	NeuralNetwork nn = getNeuralNetwork();

	OutputErrorDerivative d = getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE);
	Matrix outputErrorDerivative = d.getOutputErrorDerivative(activations.getValues(nn.getOutputLayer()), target);
	backpropagation.addValues(nn.getOutputLayer(), outputErrorDerivative);
	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(nn.getOutputLayer());
	BackPropagationLayerCalculator blc = getBPLayerCalculator();
	blc.backpropagate(nn, calculatedLayers, activations, backpropagation);
    }

    public BackPropagationLayerCalculator getBPLayerCalculator() {
	return getProperties().getParameter(Constants.BACKPROPAGATION);
    }

    public void setBPLayerCalculator(BackPropagationLayerCalculator bplc) {
	getProperties().setParameter(Constants.BACKPROPAGATION, bplc);
    }
}
