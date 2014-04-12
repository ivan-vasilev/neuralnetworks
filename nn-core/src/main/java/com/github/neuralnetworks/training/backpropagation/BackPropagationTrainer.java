package com.github.neuralnetworks.training.backpropagation;

import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Tensor;
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

    private ValuesProvider activations;
    private ValuesProvider backpropagation;

    public BackPropagationTrainer(Properties properties) {
	super(properties);
	activations = Environment.getInstance().getValuesProvider(getNeuralNetwork());
	backpropagation = Environment.getInstance().getValuesProvider(getNeuralNetwork());
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

    public void propagateForward(Tensor input) {
	NeuralNetwork nn = getNeuralNetwork();
	Set<Layer> calculatedLayers = new UniqueList<Layer>();
	calculatedLayers.add(nn.getInputLayer());
	activations.replace(nn.getInputLayer(), input);
	nn.getLayerCalculator().calculate(nn, nn.getOutputLayer(), calculatedLayers, activations);
    }

    public void propagateBackward(Tensor target) {
	NeuralNetwork nn = getNeuralNetwork();

	backpropagation.setMiniBatchSize(target.getDimensionElementsDistance(target.getDimensions().length - 1));
	OutputErrorDerivative d = getProperties().getParameter(Constants.OUTPUT_ERROR_DERIVATIVE);
	d.getOutputErrorDerivative(activations.getValues(nn.getOutputLayer()), target, backpropagation.getValues(nn.getOutputLayer()));
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
