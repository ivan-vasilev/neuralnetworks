package com.github.neuralnetworks.training.events;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Time/error log
 */
public class LogTrainingListener implements TrainingEventListener
{
	private static final Logger logger = LoggerFactory.getLogger(LogTrainingListener.class);

	private static final long serialVersionUID = 1L;

	private static final DecimalFormat df;

	static
	{
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols();
		otherSymbols.setDecimalSeparator('.');
		df = new DecimalFormat("#.##", otherSymbols);
	}

	private String name;
	private long startTime;
	private long finishTime;
	private long miniBatchTime;
	private long miniBatchTotalTime;
	private long firstMiniBatchFinishTime;
	private long lastMiniBatchFinishTime;
	private long lastLoggedMiniBatch;
	private long openclStartTime;
	private long openclCurrentTime;
	private long populateInputTotalTime;
	private long populateInputStartTime;
	private long logInterval;
	private float miniBatchLoss;
	private float prevMiniBatchLoss;
	private int miniBatches;
	private int phaseMiniBatches;

	/**
	 * log minibatches time
	 */
	private boolean logMiniBatches;

	/**
	 * dispaly input/target/networkOutput for each testing example
	 */
	private boolean logTestResults;

	private boolean isTraining;

	private boolean logBatchLoss;
	
	private boolean logWeights;

	private boolean logEpochs;

	public LogTrainingListener(String name)
	{
		super();
		this.name = name;
		this.logInterval = 5000;
		this.logEpochs = true;
	}

	public LogTrainingListener(String name, boolean logTestResults, boolean logMiniBatches)
	{
		super();
		this.name = name;
		this.logTestResults = logTestResults;
		this.logMiniBatches = logMiniBatches;
		this.logInterval = 5000;
		this.logEpochs = true;
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof PhaseStartedEvent)
		{
			reset();
			lastMiniBatchFinishTime = populateInputStartTime = startTime = System.currentTimeMillis();
			if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
			{
				openclStartTime = OpenCLCore.getInstance().getKernelsRunTime();
			}

			isTraining = false;
			if (event instanceof TrainingStartedEvent)
			{
				isTraining = true;
				logger.info("TRAINING " + name + "...");
			} else if (event instanceof TestingStartedEvent)
			{
				logger.info("TESTING " + name + "...");
			} else if (event instanceof ValidationStartedEvent)
			{
				logger.info("VALIDATING " + name + "...");
			} else
			{
				logger.info("CALCULATING " + name + "...");
			}
		} else if (event instanceof PhaseFinishedEvent)
		{
			finishTime = System.currentTimeMillis();
			String s = System.getProperty("line.separator");

			StringBuilder sb = new StringBuilder();
			sb.append("TOTAL: " + df.format((finishTime - startTime) / 1000f) + " s  total time; ");
			sb.append(df.format(miniBatchTotalTime / (miniBatches * 1000f)) + " s per minibatch of " + miniBatches + " batches");

			// log ocl time
			if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
			{
				sb.append("; OPENCL: " + df.format(openclCurrentTime / 1000000f) + " s (" + openclCurrentTime / 1000 + " ms)");
			}
			sb.append("; INPUT: " + df.format(populateInputTotalTime / 1000f) + " s, " + populateInputTotalTime + " ms");

			logger.info(sb.toString());

			if (event instanceof TestingFinishedEvent || event instanceof ValidationFinishedEvent)
			{
				NetworkCalculator<?> nc = (NetworkCalculator<?>) event.getSource();
				OutputError oe = nc.getOutputError();
				logger.info("Error: " + oe.getTotalErrorSamples() + "/" + oe.getTotalInputSize() + " samples (" + oe.getString() + ", " + df.format(oe.getTotalNetworkError() * 100) + "%)" + s + s);
			}

			if (event instanceof TrainingFinishedEvent && logWeights) 
			{
				TrainingFinishedEvent e = (TrainingFinishedEvent) event;
				Trainer<?> t = (Trainer<?>) e.getSource();
				logger.info("NETWORK WEIGHTS" + s);
				logger.info(Util.networkWeights(t.getNeuralNetwork()));
			}
		} else if (event instanceof EpochFinishedEvent && isTraining && logEpochs)
		{
			phaseMiniBatches = 0;
			logger.info("FINISH EPOCH " + ((EpochFinishedEvent) event).getBatchCount());
		}
		else if (event instanceof NewInputEvent)
		{
			populateInputStartTime = System.currentTimeMillis();
		} else if (event instanceof MiniBatchStartedEvent)
		{
			populateInputTotalTime += System.currentTimeMillis() - populateInputStartTime;
		} else if (event instanceof MiniBatchFinishedEvent)
		{
			MiniBatchFinishedEvent mbe = (MiniBatchFinishedEvent) event;

			String s = System.getProperty("line.separator");

			miniBatches++;
			phaseMiniBatches++;

			if (miniBatches == 1)
			{
				if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
				{
					StringBuilder sbDevice = new StringBuilder();
					OpenCLCore.getInstance().getMemoryUsage().entrySet().forEach(e -> {
						sbDevice.append("DEVICE ")
								.append(e.getKey())
								.append(" - ")
								.append(OpenCLCore.getInstance().getDeviceName(e.getKey()))
								.append(" (")
								.append(OpenCLCore.getInstance().getPlatformName(e.getKey()))
								.append(") - MEMORY " + df.format(e.getValue() / (1024 * 1024 * 1024f)) + " GB; ");
					});

					logger.info(sbDevice.toString());
				}

				firstMiniBatchFinishTime = System.currentTimeMillis() - startTime;
				logger.info("TOTAL: 1 batch in " + firstMiniBatchFinishTime + " ms");
			}

			miniBatchTime += System.currentTimeMillis() - lastMiniBatchFinishTime;
			miniBatchTotalTime += System.currentTimeMillis() - lastMiniBatchFinishTime;
			lastMiniBatchFinishTime = System.currentTimeMillis();

			// not very nice
			if (isTraining && logBatchLoss && mbe.getSource() instanceof BackPropagationTrainer)
			{
				BackPropagationTrainer<?> t = (BackPropagationTrainer<?>) mbe.getSource();
				Tensor networkOutput = t.getCurrentNetworkOutput();
				if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
				{
					OpenCLArrayReferenceManager.getInstance().pushToHost(networkOutput);
				}

				miniBatchLoss += t.getLossFunctionCurrentValue();
			}

			if (miniBatches > 1 && miniBatchTime / logInterval > 0 && (logMiniBatches || (!isTraining && logTestResults)))
			{
				int totalSamples = mbe.getData().getInput().getDimensions()[0] * miniBatches;
				int phaseSamples = mbe.getData().getInput().getDimensions()[0] * phaseMiniBatches;

				StringBuilder sb = new StringBuilder();

				sb.append("TOTAL: " + miniBatches + " batches (" + phaseSamples + "/" + mbe.getInputProvider().getInputSize() + " samples, " + df.format((100 * phaseSamples / ((double) mbe.getInputProvider().getInputSize()))) + "%) in " + miniBatchTotalTime + " ms, " + df.format(miniBatchTotalTime / ((float) totalSamples)) + " ms per sample");
				if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
				{
					openclCurrentTime = OpenCLCore.getInstance().getKernelsRunTime() - openclStartTime;
					sb.append("; OPENCL: " + df.format(openclCurrentTime / 1000) + " ms, "
							+ df.format(openclCurrentTime / ((float) totalSamples * 1000)) + " ms per sample");
				}
				sb.append("; INPUT: " + populateInputTotalTime + " ms, " + df.format(populateInputTotalTime / ((float) miniBatches * mbe.getData().getInput().getDimensions()[0])) + " ms per sample");

				miniBatchTime = 0;

				if (isTraining && logBatchLoss)
				{
					prevMiniBatchLoss = miniBatchLoss / (mbe.getData().getInput().getDimensions()[0] * ((float) miniBatches - lastLoggedMiniBatch));

					DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols();
					otherSymbols.setDecimalSeparator('.');

					sb.append("; LOSS: " + prevMiniBatchLoss);
					miniBatchLoss = 0;
				}

				logger.info(sb.toString());

				lastLoggedMiniBatch = miniBatches;
			}

			// log test results
			if (!isTraining && logTestResults)
			{
				if (mbe.getResults() != null)
				{
					Matrix input = (Matrix) mbe.getData().getInput();
					Matrix target = (Matrix) mbe.getData().getTarget();
					Trainer<?> t = (Trainer<?>) mbe.getSource();
					Matrix networkOutput = (Matrix) mbe.getResults().get(t.getNeuralNetwork().getOutputLayer());

					StringBuilder sb = new StringBuilder();

					for (int i = 0; i < input.getRows(); i++)
					{
						sb.append(s);
						sb.append("Input:  ");

						for (int j = 0; j < input.getColumns(); j++)
						{
							sb.append(input.get(i, j)).append("  ");
						}

						sb.append(s);
						sb.append("Output: ");
						for (int j = 0; j < networkOutput.getColumns(); j++)
						{
							sb.append(networkOutput.get(i, j)).append("  ");
						}

						sb.append(s);
						sb.append("Target: ");
						for (int j = 0; j < target.getColumns(); j++)
						{
							sb.append(target.get(i, j)).append("  ");
						}
					}
					sb.append(s);

					logger.info(sb.toString());
				}
			}

			if (isTraining && logWeights && event.getSource() instanceof BackPropagationTrainer )
			{
				BackPropagationTrainer<?> t = (BackPropagationTrainer<?>) event.getSource();
				logger.info("WEIGHT UPDATES");
				logger.info(Util.weightUpdates(t.getNeuralNetwork(), (ValuesProvider) t.getProperties().get(Constants.WEIGHT_UDPATES)));
			}
		} else if (event instanceof TestingFinishedEvent)
		{
			miniBatches++;
		}
	}

	public boolean getLogBatchLoss()
	{
		return logBatchLoss;
	}

	public void setLogBatchLoss(boolean logBatchLoss)
	{
		this.logBatchLoss = logBatchLoss;
	}

	public long getLogInterval()
	{
		return logInterval;
	}

	public void setLogInterval(long logInterval)
	{
		this.logInterval = logInterval;
	}

	public boolean getLogWeights()
	{
		return logWeights;
	}

	public void setLogWeights(boolean logWeights)
	{
		this.logWeights = logWeights;
	}

	public boolean getLogEpochs()
	{
		return logEpochs;
	}

	public void setLogEpochs(boolean logEpochs)
	{
		this.logEpochs = logEpochs;
	}

	private void reset()
	{
		firstMiniBatchFinishTime = openclStartTime = openclCurrentTime = lastLoggedMiniBatch = startTime = finishTime = miniBatchTotalTime = lastMiniBatchFinishTime = populateInputTotalTime = populateInputStartTime = miniBatches = 0;
		miniBatchLoss = phaseMiniBatches = 0;
	}
}
