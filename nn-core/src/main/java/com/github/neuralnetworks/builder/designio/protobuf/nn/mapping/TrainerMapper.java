package com.github.neuralnetworks.builder.designio.protobuf.nn.mapping;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.protobuf.nn.NNProtoBufWrapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.TrainerProtoBufWrapper;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.events.LinearLearnRateDecrListener;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.events.NetworkSaveListener;
import com.github.neuralnetworks.training.events.ValidationListener;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * @author tmey
 */
public class TrainerMapper
{

	public static NeuralNetworkBuilder mapProtoBufTrainConfigToBuilder(TrainerProtoBufWrapper.TrainParameter solverParameter)
	{
		return mapProtoBufTrainConfigToBuilder(new NeuralNetworkBuilder(), solverParameter);
	}

	/**
	 * map the general parameter
	 *
	 * @param neuralNetworkBuilder
	 * @param solverParameter
	 * @return
	 */
	public static NeuralNetworkBuilder mapProtoBufTrainConfigToBuilder(NeuralNetworkBuilder neuralNetworkBuilder, TrainerProtoBufWrapper.TrainParameter solverParameter)
	{
		if (neuralNetworkBuilder == null)
		{
			throw new IllegalArgumentException("neuralNetworkBuilder must be not null!");
		}

		if (solverParameter.hasBatchSize())
		{
			neuralNetworkBuilder.setTrainingBatchSize(solverParameter.getBatchSize());
		}
		if (solverParameter.hasMaxEpochs())
		{
			neuralNetworkBuilder.setTrainingBatchSize(solverParameter.getMaxEpochs());
		}

		if (solverParameter.hasLearnParam())
		{
			NNProtoBufWrapper.LearnParameter learnParam = solverParameter.getLearnParam();
			if (learnParam.hasLr())
			{
				neuralNetworkBuilder.setLearningRate(learnParam.getLr());
			}
			if (learnParam.hasMomentum())
			{
				neuralNetworkBuilder.setMomentum(learnParam.getMomentum());
			}
			if (learnParam.hasWeightDecay())
			{
				neuralNetworkBuilder.setL1weightDecay(learnParam.getWeightDecay());
			}
		}

		if (solverParameter.hasTestParam())
		{
			TrainerProtoBufWrapper.TestParameter testParam = solverParameter.getTestParam();

			if (!testParam.hasTestInput())
			{
				throw new IllegalArgumentException("The test requires an input!");
			}
			neuralNetworkBuilder.setTestingSet(InputMapper.parseInput(testParam.getTestInput()));

			if (testParam.hasBatchSize())
			{
				neuralNetworkBuilder.setTestBatchSize(testParam.getBatchSize());
			}
		}

		if (solverParameter.hasLoggingParam())
		{
			TrainerProtoBufWrapper.LoggingParameter loggingParam = solverParameter.getLoggingParam();

			LogTrainingListener logTrainingListener = new LogTrainingListener("nn", false, true);

			if (loggingParam.hasInterval())
			{
				logTrainingListener.setLogInterval(loggingParam.getInterval() * 1000);
			}
			if (loggingParam.hasBatchLoss())
			{
				logTrainingListener.setLogBatchLoss(loggingParam.getBatchLoss());
			}
			if (loggingParam.hasEpoch())
			{
				logTrainingListener.setLogEpochs(loggingParam.getEpoch());
			}
			if (loggingParam.hasWeights())
			{
				logTrainingListener.setLogWeights(loggingParam.getWeights());
			}

			neuralNetworkBuilder.addEventListener(logTrainingListener);
		}

		if (solverParameter.hasValidationParam())
		{
			TrainerProtoBufWrapper.ValidationParameter validationParam = solverParameter.getValidationParam();

			if (!validationParam.hasValidationInput())
			{
				throw new IllegalArgumentException("The validation listener needs input data!");
			}

			TrainingInputProvider validationInput = InputMapper.parseInput(validationParam.getValidationInput());


			ValidationListener validationListener = new ValidationListener(validationInput, validationParam.getAcceptanceError(), validationParam.getTestInterval());

			neuralNetworkBuilder.addEventListener(validationListener);
		}

		if (solverParameter.hasSnapshotParam())
		{
			TrainerProtoBufWrapper.SnapShotParameter snapshotParam = solverParameter.getSnapshotParam();

			if (snapshotParam.hasPath())
			{
				throw new IllegalArgumentException("A path/file is required to save the snapshots and the final trained network!");
			}

			NetworkSaveListener networkSaveListener = new NetworkSaveListener(snapshotParam.getPath(), snapshotParam.getSnapshotDuringTrain());
			networkSaveListener.setEpochInterval(snapshotParam.getInterval());

			neuralNetworkBuilder.addEventListener(networkSaveListener);
		}


		if (solverParameter.hasRuntimeParam())
		{
			TrainerProtoBufWrapper.RuntimeOptions runtimeParam = solverParameter.getRuntimeParam();

			RuntimeConfiguration runtimeConfiguration = new RuntimeConfiguration();

			if (runtimeParam.hasDeviceId())
			{
				runtimeConfiguration.getOpenCLConfiguration().setPreferredDevice(runtimeParam.getDeviceId());
			}
			if (runtimeParam.hasPrecompilation())
			{
				runtimeConfiguration.getOpenCLConfiguration().setUseOptionsString(runtimeParam.getPrecompilation());
			}

			switch (runtimeParam.getProcessorMode())
			{
			case CPU:
				runtimeConfiguration.setCalculationProvider(RuntimeConfiguration.CalculationProvider.CPU);
				runtimeConfiguration.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
				break;
			case OPENCL:
				runtimeConfiguration.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);
				break;
			case JAVA:
				runtimeConfiguration.setCalculationProvider(RuntimeConfiguration.CalculationProvider.CPU);
				runtimeConfiguration.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
				break;
			default:
				throw new IllegalArgumentException("Unknown processor mode " + runtimeParam.getProcessorMode());
			}


			neuralNetworkBuilder.setRuntimeConfiguration(runtimeConfiguration);
		}

		if (solverParameter.hasLrDecr())
		{
			TrainerProtoBufWrapper.LearnRateDecrParam lrDecr = solverParameter.getLrDecr();

			if (!lrDecr.hasLrPolicy())
			{
				throw new IllegalArgumentException("the learning rate decreasing needs a policy!");
			}

			switch (lrDecr.getLrPolicy())
			{
			case LINEAR:
				LinearLearnRateDecrListener linearLearnRateDecrListener = new LinearLearnRateDecrListener();

				if (lrDecr.hasPower())
				{
					linearLearnRateDecrListener.setReductionFactor(lrDecr.getPower());
				}
				linearLearnRateDecrListener.setChangeInterval(lrDecr.getUpdateInterval());

				neuralNetworkBuilder.addEventListener(linearLearnRateDecrListener);
				break;
			default:
				throw new IllegalArgumentException("Unknown learning rate policy: " + lrDecr.getLrPolicy());
			}
		}

		return neuralNetworkBuilder;
	}

}
