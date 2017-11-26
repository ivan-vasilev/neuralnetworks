package com.github.neuralnetworks.training;

import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.DNN;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;

/**
 * Training Input Provider for deep network trainers
 */
public class DeepTrainerTrainingInputProvider extends TrainingInputProviderImpl
{

	private static final long serialVersionUID = 1L;

	private TrainingInputProvider inputProvider;
	private TrainingInputData inputDataBase;
	private DNN<?> dnn;
	private NeuralNetwork partialDNN;
	private NeuralNetwork currentNN;
	private Set<Layer> calculatedLayers;
	private ValuesProvider layerResults;

	public DeepTrainerTrainingInputProvider(TrainingInputProvider inputProvider, DNN<?> dnn, NeuralNetwork currentNN, int batchSize)
	{
		super();
		this.inputProvider = inputProvider;
		this.dnn = dnn;
		this.currentNN = currentNN;

		if (currentNN != dnn.getFirstNeuralNetwork()) {
			this.calculatedLayers = new HashSet<>();
			this.layerResults = TensorFactory.tensorProvider(batchSize, Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory(), dnn, currentNN);
			this.inputDataBase = new TrainingInputDataImpl(layerResults.get(dnn.getInputLayer()));
			this.partialDNN = new NeuralNetworkImpl();
			partialDNN.getLayers().addAll(dnn.getLayers());
			partialDNN.getLayers().removeIf(l -> {
				int id = dnn.getNeuralNetworks().indexOf(currentNN);
				for (int i = 0; i < id; i++) {
					if (dnn.getNeuralNetworks().get(i).getLayers().contains(l)) {
						return false;
					}
				}

				return true;
			});
			partialDNN.getLayers().add(currentNN.getInputLayer());
		}
	}

	@Override
	public void getNextInput(Tensor input)
	{
		if (dnn.getFirstNeuralNetwork() != currentNN)
		{
			inputProvider.populateNext(inputDataBase);
			calculatedLayers.clear();
			calculatedLayers.add(dnn.getInputLayer());
			dnn.getLayerCalculator().calculate(partialDNN, currentNN.getInputLayer(), calculatedLayers, layerResults);
			TensorFactory.copy(layerResults.get(currentNN.getInputLayer()), input);
		} else {
			inputProvider.getNextInput(input);
		}
	}

//	@Override
//	public float[] getNextInput() {
//		float[] result = null;
//		if (dnn.getFirstNeuralNetwork() != currentNN)
//		{
//			if (inputId == inputDataBase.getInput().getDimensions()[0] || nextInput == null) {
//				inputId = 0;
//				inputProvider.populateNext(inputDataBase);
//				calculatedLayers.clear();
//				calculatedLayers.add(dnn.getInputLayer());
//				dnn.getLayerCalculator().calculate(partialDNN, currentNN.getInputLayer(), calculatedLayers, layerResults);
//				Tensor t = layerResults.get(currentNN.getInputLayer());
//				if (nextInput == null) {
//					nextInput = new float[t.getSize() / t.getDimensions()[0]];
//				}
//			}
//
//			Tensor t = layerResults.get(currentNN.getInputLayer());
//			System.arraycopy(t.getElements(), t.getStartIndex() + inputId * nextInput.length, nextInput, 0, nextInput.length);
//
//			result = nextInput;
//		} else {
//			result = inputProvider.getNextInput();
//		}
//
//		return result;
//	}

//	@Override
//	public void afterBatch(TrainingInputData ti)
//	{
//		if (dnn.getFirstNeuralNetwork() != currentNN)
//		{
//			inputProvider.populateNext(inputDataBase);
//			calculatedLayers.clear();
//			calculatedLayers.add(dnn.getInputLayer());
//			dnn.getLayerCalculator().calculate(partialDNN, currentNN.getInputLayer(), calculatedLayers, layerResults);
//			TensorFactory.copy(layerResults.get(currentNN.getInputLayer()), ti.getInput());
//		}
//	}

	@Override
	public void getNextTarget(Tensor target)
	{
		inputProvider.getNextTarget(target);
	}

//	@Override
//	public float[] getNextTarget()
//	{
//		return inputProvider.getNextTarget();
//	}

	@Override
	public int getInputSize()
	{
		return inputProvider.getInputSize();
	}

	@Override
	public void reset()
	{
		inputProvider.reset();
	}

	public TrainingInputProvider getInputProvider()
	{
		return inputProvider;
	}

	public DNN<?> getDnn()
	{
		return dnn;
	}

	public NeuralNetwork getCurrentNN()
	{
		return currentNN;
	}

	@Override
	public int getInputDimensions()
	{
		return partialDNN.getOutputLayer().getUnitCount(partialDNN.getOutputLayer().getConnections());
	}

	@Override
	public int getTargetDimensions()
	{
		return inputProvider.getTargetDimensions();
	}
}
