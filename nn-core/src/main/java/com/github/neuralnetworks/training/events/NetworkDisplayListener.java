package com.github.neuralnetworks.training.events;

import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer.LossFunctionEvent;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * Display the values of the network
 */
public class NetworkDisplayListener implements TrainingEventListener, PropagationEventListener
{
	private static final long serialVersionUID = 1L;

	private int miniBatches;
	private int currentLayer;
	private boolean displayLayers;
	private boolean displayWeights;

	public NetworkDisplayListener()
	{
		super();
		this.displayLayers = true;
		this.displayWeights = true;
	}

	public NetworkDisplayListener(boolean displayLayers, boolean displayWeights)
	{
		super();
		this.displayLayers = displayLayers;
		this.displayWeights = displayWeights;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof MiniBatchStartedEvent)
		{
			currentLayer = 0;
			System.out.println("START MINI BATCH: " + ++miniBatches + " (" + Environment.getInstance().getRuntimeConfiguration().getCalculationProvider().toString() + ")");
			System.out.println();
			System.out.println("Input");
			MiniBatchStartedEvent mbs = (MiniBatchStartedEvent) event;
			Util.printTensor(mbs.getData().getInput().getElements(), mbs.getData().getInput().getDimensions()[0]);
		} else if (event instanceof LossFunctionEvent) 
		{
			LossFunctionEvent lfe = (LossFunctionEvent) event;

			System.out.println("Loss function");
			Util.printTensor(lfe.getResult().getElements(), lfe.getResult().getDimensions()[0]);

			System.out.println("Target");
			Util.printTensor(lfe.getTarget().getElements(), lfe.getTarget().getDimensions()[0]);
		} else if (event instanceof MiniBatchFinishedEvent) 
		{
			System.out.println("END MINI BATCH: " + miniBatches + " (" + Environment.getInstance().getRuntimeConfiguration().getCalculationProvider().toString() + ")");
		}
	}

	@Override
	public void handleEvent(PropagationEvent event)
	{
		if (event instanceof PropagationEvent)
		{
			PropagationEvent pe = event;
			if (displayLayers)
			{
				if (pe.getLayer() == pe.getNeuralNetwork().getInputLayer())
				{
					System.out.println("Input");
				} else if (pe.getLayer() == pe.getNeuralNetwork().getOutputLayer())
				{
					System.out.println("Output");
				}
				else
				{
					System.out.println("Layer " + ++currentLayer);
				}

				Tensor tensor = TensorFactory.tensor(pe.getLayer(), pe.getConnections(), pe.getResults());
				Util.printTensor(tensor.getElements(), tensor.getDimensions()[0]);
			}

			if (displayWeights)
			{
				pe.getConnections().stream().filter(c -> c instanceof WeightsConnections).forEach(c -> {
					System.out.println("Weights");
					Tensor tensor = ((WeightsConnections) c).getWeights();
					Util.printTensor(tensor.getElements(), tensor.getDimensions()[0]);
				});
			}
		}
	}
}
