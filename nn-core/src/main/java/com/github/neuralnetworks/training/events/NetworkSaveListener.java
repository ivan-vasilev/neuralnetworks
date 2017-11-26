package com.github.neuralnetworks.training.events;

import java.io.OutputStream;

import org.apache.commons.lang.StringUtils;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.util.Serializer;

/**
 * Saves the the network after each epoch. the saveEachEpoch parameter determines whether to use different files for each epoch.
 */
public class NetworkSaveListener implements TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	private String filePath;
	private OutputStream outputStream;
	private int epochInterval = 1;

	/**
	 * whether to save each epoch in separate file or overwrite the same file
	 */
	private boolean saveEachEpoch;

	/**
	 * Save trainer into file
	 * 
	 * @param filePath
	 * @param saveEachEpoch
	 */
	public NetworkSaveListener(String filePath, boolean saveEachEpoch)
	{
		if (filePath == null)
		{
			throw new IllegalArgumentException("saveFile must be not null!");
		}

		this.filePath = filePath;
		this.saveEachEpoch = saveEachEpoch;
	}

	/**
	 * Save trainer into outputStream
	 * 
	 * @param outputStream
	 */
	public NetworkSaveListener(OutputStream outputStream)
	{
		this.outputStream = outputStream;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof EpochFinishedEvent && event.getSource() instanceof NetworkCalculator<?>
				&& ((EpochFinishedEvent) event).getEpoch() % epochInterval == 0)
		{
			EpochFinishedEvent epochFinishedEvent = (EpochFinishedEvent) event;
			NetworkCalculator<?> trainer = (NetworkCalculator<?>) event.getSource();
			NeuralNetwork nn = trainer.getNeuralNetwork();

			if (!StringUtils.isEmpty(filePath))
			{
				Serializer.saveNetwork(nn, saveEachEpoch ? filePath + ".epoch_" + epochFinishedEvent.getBatchCount() : filePath);
			} else if (outputStream != null)
			{
				Serializer.saveNetwork(nn, outputStream);
			}
		}
	}

	public int getEpochInterval()
	{
		return epochInterval;
	}

	public void setEpochInterval(int epochInterval)
	{
		this.epochInterval = epochInterval;
	}
}
