package com.github.neuralnetworks.calculation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLManagementListener;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.MiniBatchStartedEvent;
import com.github.neuralnetworks.training.events.NewInputEvent;
import com.github.neuralnetworks.training.events.PhaseFinishedEvent;
import com.github.neuralnetworks.training.events.PhaseStartedEvent;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.UniqueList;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * interface that contains methods common for NN calculation
 */
public interface NetworkCalculator<N extends NeuralNetwork> extends Serializable
{
	public abstract Properties getProperties();

	public default void calculate(TrainingInputProvider inputProvider)
	{
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL
				&& getListeners().stream().filter(l -> l instanceof OpenCLManagementListener).count() == 0)
		{
			addEventListener(new OpenCLManagementListener(), 0);
		}

		triggerEvent(phaseStartedEvent());

		N n = getNeuralNetwork();

		if (inputProvider != null && n != null && n.getLayerCalculator() != null)
		{
			setSkipCurrentMiniBatch(false);

			inputProvider.reset();

			Set<Layer> calculatedLayers = new UniqueList<>();
			ValuesProvider results = TensorFactory.tensorProvider(n, getTestBatchSize(), Environment.getInstance().getRuntimeConfiguration().getUseDataSharedMemory());

			OutputError oe = getOutputError();
			if (oe != null)
			{
				oe.reset();
				results.add(oe, results.get(n.getOutputLayer()).getDimensions());
			}

			TrainingInputData input = new TrainingInputDataImpl(results.get(n.getInputLayer()), results.get(oe));
			for (int i = 0; i < inputProvider.getInputSize(); i += getTestBatchSize())
			{
				triggerEvent(new NewInputEvent(this, input, results, i));

				inputProvider.populateNext(input);

				triggerEvent(new MiniBatchStartedEvent(this, input, results, i));

				if (!getSkipCurrentMiniBatch())
				{
					calculatedLayers.clear();
					calculatedLayers.add(n.getInputLayer());
					n.getLayerCalculator().calculate(n, n.getOutputLayer(), calculatedLayers, results);
				} else
				{
					setSkipCurrentMiniBatch(false);
				}

				triggerEvent(new MiniBatchFinishedEvent(this, input, results, i / getTestBatchSize(), inputProvider));

				if (oe != null)
				{
					oe.addItem(results.get(n.getOutputLayer()), input.getTarget());
				}
			}
		}

		triggerEvent(phaseFinishedEvent());
	}

	public default PhaseStartedEvent phaseStartedEvent()
	{
		return new PhaseStartedEvent(this);
	}
	
	public default PhaseFinishedEvent phaseFinishedEvent()
	{
		return new PhaseFinishedEvent(this);
	}

	public void setSkipCurrentMiniBatch(boolean skipCurrentMiniBatch);

	public boolean getSkipCurrentMiniBatch();

	public default N getNeuralNetwork()
	{
		return getProperties().getParameter(Constants.NEURAL_NETWORK);
	}

	public default void setNeuralNetwork(N neuralNetwork)
	{
		getProperties().setParameter(Constants.NEURAL_NETWORK, neuralNetwork);
	}

	public default OutputError getOutputError()
	{
		return getProperties().getParameter(Constants.OUTPUT_ERROR);
	}

	public default void setOutputError(OutputError outputError)
	{
		getProperties().setParameter(Constants.OUTPUT_ERROR, outputError);
	}

	public default List<TrainingEventListener> getListeners()
	{
		return getProperties().getParameter(Constants.LISTENERS);
	}

	public default void setListeners(List<TrainingEventListener> listeners)
	{
		getProperties().setParameter(Constants.LISTENERS, listeners);
	}

	public default Integer getTestBatchSize()
	{
		return getProperties().getParameter(Constants.TEST_BATCH_SIZE) != null ? getProperties().getParameter(Constants.TEST_BATCH_SIZE) : 1;
	}

	public default void setTestBatchSize(int batchSize)
	{
		getProperties().setParameter(Constants.TEST_BATCH_SIZE, batchSize);
	}

	/**
	 * add event listener on position
	 * 
	 * @param listener
	 * @param position
	 */
	public default void addEventListener(TrainingEventListener listener, int position)
	{
		if (getListeners() == null)
		{
			setListeners(new UniqueList<>());
		}

		getListeners().add(position, listener);
	}

	public default void addEventListener(TrainingEventListener listener)
	{
		if (getListeners() == null)
		{
			setListeners(new UniqueList<>());
		}

		getListeners().add(listener);
	}

	public default void removeEventListener(TrainingEventListener listener)
	{
		if (getListeners() != null)
		{
			getListeners().remove(listener);
		}
	}

	public default void triggerEvent(TrainingEvent event)
	{
		if (getListeners() != null)
		{
			List<TrainingEventListener> listeners = new ArrayList<>(getListeners());
			listeners.stream().filter(l -> event.getSource() != l).forEach(l -> l.handleEvent(event));
		}
	}
}
