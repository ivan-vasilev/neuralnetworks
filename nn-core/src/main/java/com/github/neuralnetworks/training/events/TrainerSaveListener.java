package com.github.neuralnetworks.training.events;

import java.io.OutputStream;

import org.apache.commons.lang.StringUtils;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.util.Serializer;

/**
 * Saves the trainer and the network after each epoch. the saveEachEpoch parameter determines whether to use different files for each epoch. 
 * 
 * @author tmey
 */
public class TrainerSaveListener implements TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	private String filePath;
	private OutputStream outputStream;

	/**
	 * whether to save each epoch in separate file or overwrite the same file
	 */
	private boolean saveEachEpoch;

	/**
	 * Save trainer into file
	 * @param filePath
	 * @param saveEachEpoch
	 */
	public TrainerSaveListener(String filePath, boolean saveEachEpoch)
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
	 * @param outputStream
	 */
	public TrainerSaveListener(OutputStream outputStream)
	{
		this.outputStream = outputStream;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof EpochFinishedEvent && event.getSource() instanceof Trainer)
		{
			EpochFinishedEvent epochFinishedEvent = (EpochFinishedEvent) event;
			if (!StringUtils.isEmpty(filePath))
			{
				Serializer.saveTrainer((Trainer<?>) event.getSource(), saveEachEpoch ? filePath + ".epoch_" + epochFinishedEvent.getBatchCount() : filePath);
			} else if (outputStream != null)
			{
				Serializer.saveTrainer((Trainer<?>) event.getSource(), outputStream);
			}
		}
	}
}
