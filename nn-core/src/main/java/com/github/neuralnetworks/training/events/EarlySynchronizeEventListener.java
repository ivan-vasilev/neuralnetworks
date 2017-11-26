package com.github.neuralnetworks.training.events;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.analyzing.ConnectionAnalysis;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.util.Environment;

/**
 * @author tmey
 */
public class EarlySynchronizeEventListener implements TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	private static final Logger logger = LoggerFactory.getLogger(EarlySynchronizeEventListener.class);

	private File saveFile = null;
	private Trainer<?> trainer;
	private boolean saveTrainerInsteadOfNetwork = false;

	private int sampleStep = -1;

	private String lastWorkingWeights = null;

	public EarlySynchronizeEventListener(Trainer<?> trainer)
	{
		if (trainer == null)
		{
			throw new IllegalArgumentException("trainer must be not null!");
		}

		this.trainer = trainer;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{


		if ((event instanceof MiniBatchFinishedEvent && sampleStep > 0 && ((MiniBatchFinishedEvent) event).getBatchCount() % sampleStep == 0)
				|| (event instanceof EpochFinishedEvent || event instanceof TrainingFinishedEvent))
		{
			// synchronize

			if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration() != null)
			{
				OpenCLArrayReferenceManager.getInstance().pushAllToHost();
			}

			// analyze
			if (sampleStep <= 0)
			{
				logger.info(ConnectionAnalysis.analyseConnectionWeights((com.github.neuralnetworks.architecture.NeuralNetworkImpl) trainer.getNeuralNetwork()));
			} else
			{
				// to find bugs

				String currentResult = ConnectionAnalysis.analyseConnectionWeights((com.github.neuralnetworks.architecture.NeuralNetworkImpl) trainer.getNeuralNetwork());

				if (event instanceof MiniBatchFinishedEvent)
				{
					currentResult = "batch count " + ((MiniBatchFinishedEvent) event).getBatchCount() + "\n" + currentResult;
				}

				if (event instanceof EpochFinishedEvent)
				{
					logger.info(ConnectionAnalysis.analyseConnectionWeights((com.github.neuralnetworks.architecture.NeuralNetworkImpl) trainer.getNeuralNetwork()));
				}

				if (currentResult.contains("NaN"))
				{
					logger.warn("last working weights:\n\n" + lastWorkingWeights + "\n\ncurrent weights:\n\n" + currentResult);
				} else
				{
					this.lastWorkingWeights = currentResult;
				}
			}

			// save
			if (saveFile != null)
			{

				try
				{
					if (saveFile.getParent() != null || !saveFile.getParent().isEmpty())
					{
						saveFile.getParentFile().mkdirs();
					}

					ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(saveFile));
					if (saveTrainerInsteadOfNetwork)
					{
						TrainingInputProvider testingInputProvider = trainer.getTestingInputProvider();
						TrainingInputProvider trainingInputProvider = trainer.getTrainingInputProvider();

						trainer.setTestingInputProvider(null);
						trainer.setTrainingInputProvider(null);

						trainer.setTestingInputProvider(testingInputProvider);
						trainer.setTrainingInputProvider(trainingInputProvider);

						outputStream.writeObject(trainer);
					} else
					{
						outputStream.writeObject(trainer.getNeuralNetwork());
					}
					outputStream.flush();
					outputStream.close();
				} catch (Exception e)
				{
					logger.warn("can't save the network", e);
				}

			}
		}
	}

	public File getSaveFile()
	{
		return saveFile;
	}

	public void setSaveFile(File saveFile)
	{
		this.saveFile = saveFile;
	}

	public int getSampleStep()
	{
		return sampleStep;
	}

	public void setSampleStep(int sampleStep)
	{
		this.sampleStep = sampleStep;
	}
}
