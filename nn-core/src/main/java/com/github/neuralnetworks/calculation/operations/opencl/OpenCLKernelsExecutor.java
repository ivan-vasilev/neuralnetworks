package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLAveragePooling2D;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLClear;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLConv2DBPWeightUpdates;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLConv2DFF;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLFill;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLFullyConnectedWeightUpdates;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLLRN;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMSE;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMask;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMaxPooling2D;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLNoise;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLNoiseMask;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLReLU;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLReLUDerivative;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLSigmoid;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLSigmoidDerivative;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLSoftReLU;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLSoftReLUDerivative;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLSoftmax;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLSoftmaxLoss;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLStochasticPooling2D;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLTanh;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLTanhDerivative;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLWeightedSum;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLAveragePooling2DBP.OpenCLAveragePooling2DBPCC;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLConv2DBP.OpenCLConv2DBPCC;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLLRNBP.OpenCLLRNBPCC;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMaxPooling2DBP.OpenCLMaxPooling2DBPCC;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLWeightedSumBP.OpenCLWeightedSumBPCC;
import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.training.events.EpochFinishedEvent;
import com.github.neuralnetworks.training.events.MiniBatchFinishedEvent;
import com.github.neuralnetworks.training.events.MiniBatchStartedEvent;
import com.github.neuralnetworks.training.events.PhaseFinishedEvent;
import com.github.neuralnetworks.training.events.PhaseStartedEvent;
import com.github.neuralnetworks.training.events.TestingStartedEvent;
import com.github.neuralnetworks.training.events.ValidationStartedEvent;
import com.github.neuralnetworks.util.Environment;

/**
 * Kernel executor that takes care of method execution
 */
public class OpenCLKernelsExecutor implements TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	private static OpenCLKernelsExecutor singleton = new OpenCLKernelsExecutor();

	private transient List<OpenCLKernelData> kernels;

	private transient Set<int[]> cloneReferences;

	private transient char[] jobs;

	private boolean isTesting;

	private boolean isRunning;

	private OpenCLKernelsExecutor()
	{
		super();
		init();
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		init();
	}

	private void init()
	{
		this.kernels = new ArrayList<>();
		this.cloneReferences = new HashSet<>();
	}

	public static OpenCLKernelsExecutor getInstance()
	{
		return singleton;
	}

	public void execute(OpenCLKernelData kernel)
	{
		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations())
		{
			if (!isRunning)
			{
				throw new RuntimeException("PhaseStartedEvent not fired");
			}

			if (!kernels.contains(kernel))
			{
				kernels.add(kernel);
			}
		} else
		{
			List<OpenCLKernelData> kernels = Arrays.asList(new OpenCLKernelData[] { kernel });
			char[] jobs = prepareJobs(kernels);
			pushArraysToDevice();
			modifyArraysOnDevice(kernels);
			OpenCLCore.getInstance().kernelRunJob(jobs);

			if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getSynchronizeAfterOpertation())
			{
				OpenCLArrayReferenceManager.getInstance().pushAllToHost();
			}
		}
	}

	private void modifyArraysOnDevice(List<OpenCLKernelData> kernels) {
		OpenCLKernelReferenceManager km = OpenCLKernelReferenceManager.getInstance();

		for (OpenCLKernelData kd : kernels)
		{
			OpenCLKernelReference ref = km.get(kd);
			if (ref.getModifiedArrays() != null)
			{
				for (float[] a : ref.getModifiedArrays())
				{
					OpenCLArrayReference r = OpenCLArrayReferenceManager.getInstance().getArrayReference(a, ref.getDeviceId());
					if (r != null)
					{
						r.setIsModifiedOnDevice(true);
					}
				}
			}
		}
	}

	private void pushArraysToDevice()
	{
		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPushToDeviceBeforeOperation())
		{
			OpenCLArrayReferenceManager.getInstance().pushAllToDevice();
		}
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
		if (event instanceof PhaseStartedEvent)
		{
			endPhase();

			kernels.clear();
			jobs = null;
			isTesting = event instanceof TestingStartedEvent || event instanceof ValidationStartedEvent;
			isRunning = true;
		} else if (event instanceof PhaseFinishedEvent)
		{
			endPhase();
		} else if (event instanceof EpochFinishedEvent)
		{
			OpenCLArrayReferenceManager.getInstance().pushAllToHost();
		} else if (event instanceof MiniBatchFinishedEvent && Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations())
		{
			if (jobs == null)
			{
				jobs = prepareJobs(kernels);
			}

			pushArraysToDevice();
			modifyArraysOnDevice(kernels);

			OpenCLCore.getInstance().kernelRunJob(jobs);

			if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getSynchronizeAfterOpertation())
			{
				OpenCLArrayReferenceManager.getInstance().pushAllToHost();
			} else if (isTesting)
			{
				MiniBatchFinishedEvent mbe = (MiniBatchFinishedEvent) event;
				NetworkCalculator<?> nc = (NetworkCalculator<?>) event.getSource();
				float[] output = mbe.getResults().get(nc.getNeuralNetwork().getOutputLayer()).getElements();
				OpenCLArrayReferenceManager.getInstance().pushToHost(output);
			}
		} else if (event instanceof MiniBatchStartedEvent)
		{
			MiniBatchStartedEvent mbe = (MiniBatchStartedEvent) event;
			if (mbe.getData().getInput() != null)
			{
				OpenCLArrayReferenceManager.getInstance().pushToDevice(mbe.getData().getInput().getElements());
			}

			if (mbe.getData().getTarget() != null)
			{
				OpenCLArrayReferenceManager.getInstance().pushToDevice(mbe.getData().getTarget().getElements());
			}

			if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getAggregateOperations() && mbe.getSource() instanceof NetworkCalculator<?>)
			{
				NetworkCalculator<?> t = (NetworkCalculator<?>) mbe.getSource();
				if (jobs != null)
				{
					t.setSkipCurrentMiniBatch(true);;
				}
			}
		}
	}

	private void endPhase()
	{
		if (isRunning)
		{
			kernels.clear();
			isRunning = false;
			jobs = null;
			OpenCLArrayReferenceManager.getInstance().pushAllToHost();
			if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getFinalyzeDeviceAfterPhase())
			{
				OpenCLKernelReferenceManager.getInstance().getKernelReferences().entrySet().stream().filter(e -> e.getKey() instanceof OpenCLKernelData)
						.forEach(e -> ((OpenCLKernelData) e.getKey()).destroyKernel());
				OpenCLCore.getInstance().finalizeDeviceAll();

				if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getRestartLibraryAfterPhase())
				{
					OpenCLCore.getInstance().restartLibrary();
				}
			}
		}
	}

	public char[] prepareJobs(List<OpenCLKernelData> kernels)
	{
		char[] result = null;
		if (kernels.size() > 0)
		{
			prepareKernels(kernels);

			StringBuilder sb = new StringBuilder();
			Set<Integer> blocked = new HashSet<>();

			OpenCLKernelReferenceManager km = OpenCLKernelReferenceManager.getInstance();

			// build kernels
			kernels.stream().filter(k -> km.get(k) == null).forEach(k -> km.put(k, k.createKernel()));

			for (OpenCLKernelData kd : kernels)
			{
				OpenCLKernelReference ref = km.get(kd);

				// find input reference
				List<OpenCLKernelData> prev = new ArrayList<>();
				for (int i = 0; i < kernels.indexOf(kd); i++) {
					OpenCLKernelData k = kernels.get(i);
					Tensor kdin = kd.getInput();
					Tensor kdout = kd.getOutput();
					//Tensor kin = k.getInput();
					Tensor kout = k.getOutput();
					if (((kdin != null && kout != null) && (kdin == kout || (kdin.getElements() == kout.getElements() && kdin.getEndIndex() >= kout.getStartIndex() && kdin.getStartIndex() <= kout.getEndIndex()))) ||
						((kdout != null && kout != null) && (kdout == kout || (kdout.getElements() == kout.getElements() && kdout.getEndIndex() >= kout.getStartIndex() && kdout.getStartIndex() <= kout.getEndIndex()))))
					{
						OpenCLKernelReference prevRef = km.get(k);
						if (!blocked.contains(prevRef.getId()))
						{
							sb.append("!").append(prevRef.getId()).append(" ");
							blocked.add(prevRef.getId());
						}

						prev.add(k);
					}
				}

				for (OpenCLKernelData p : prev)
				{
					// add blocking entry
					OpenCLKernelReference prevRef = km.get(p);
					if (!blocked.contains(prevRef.getId()))
					{
						sb.append("!").append(prevRef.getId()).append(" ");
						blocked.add(prevRef.getId());
					}

					// clone if necessary
					int idid = km.get(p).getDeviceId(), odid = ref.getDeviceId();
					if (idid != odid)
					{
						int[] clone = cloneReferences.stream().filter(r -> r[0] == idid && r[1] == odid).findFirst().orElse(null);

						OpenCLKernelReference cloneRef = null;
						if (clone == null || (cloneRef = km.get(clone)) == null)
						{
							int cloneId = OpenCLCore.getInstance().cloneFloatBuf(OpenCLArrayReferenceManager.getInstance().getArrayReference(p.getOutput(), idid).getId(), odid);
							cloneRef = new OpenCLKernelReference(null, cloneId, null);
							if (clone == null)
							{
								clone = new int[] { idid, odid };
								km.put(clone, cloneRef);
							}

							sb.append(cloneRef.getId()).append(" ");
						}

						break;
					}
				}

				sb.append(ref.getId()).append(" ");
				if ((kd instanceof OpenCLClear || kd instanceof OpenCLLossFunction))
				{
					sb.append("!").append(ref.getId()).append(" ");
					blocked.add(ref.getId());
				}
			}

			kernels.stream().filter(k -> !blocked.contains(km.get(k).getId())).forEach(k -> sb.append("!").append(km.get(k).getId()).append(" "));

			result = sb.toString().trim().toCharArray();
		}

		return result;
	}

	public void prepareKernels(List<OpenCLKernelData> kernels)
	{
		OpenCLKernelReferenceManager km = OpenCLKernelReferenceManager.getInstance();

		if (Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getUseOptionsString())
		{
			Set<Integer> devices = new HashSet<>();
			kernels.forEach(k -> devices.add(k.getDeviceId()));
			StringBuilder deviceOptions = new StringBuilder();

			Map<Class<?>, Integer> occurences = new HashMap<>();

			for (int i = 0; i < kernels.size(); i++)
			{
				OpenCLKernelData kd = kernels.get(i);

				if (OpenCLWeightedSum.class.isAssignableFrom(kd.getClass()))
				{
					if (occurences.containsKey(OpenCLWeightedSum.class))
					{
						occurences.put(OpenCLWeightedSum.class, occurences.get(OpenCLWeightedSum.class) + 1);
					} else
					{
						occurences.put(OpenCLWeightedSum.class, 1);
					}

					String kernelOptions = kd.kernelOptions(occurences.get(OpenCLWeightedSum.class));
					if (kernelOptions != null)
					{
						deviceOptions.append(kernelOptions).append(" ");
					}
				} else
				{
					if (occurences.containsKey(kd.getClass()))
					{
						occurences.put(kd.getClass(), occurences.get(kd.getClass()) + 1);
					} else
					{
						occurences.put(kd.getClass(), 1);
					}

					String kernelOptions = kd.kernelOptions(occurences.get(kd.getClass()));
					if (kernelOptions != null)
					{
						deviceOptions.append(kernelOptions).append(" ");
					}
				}
			}

			occurences.entrySet().forEach(e -> {
				if (e.getKey() == OpenCLWeightedSum.class || e.getKey() == OpenCLWeightedSumBPCC.class)
				{
					OpenCLCore.getInstance().cloneWeightedSum(e.getValue());
				} else if (e.getKey() == OpenCLConv2DFF.class) 
				{
					OpenCLCore.getInstance().cloneConv2DFF(e.getValue());
				} else if (e.getKey() == OpenCLLRN.class) 
				{
					OpenCLCore.getInstance().cloneLRN(e.getValue());
				} else if (e.getKey() == OpenCLConv2DBPCC.class) 
				{
					OpenCLCore.getInstance().cloneBackpropagationConv2D2(e.getValue());
				} else if (e.getKey() == OpenCLMaxPooling2D.class) 
				{
					OpenCLCore.getInstance().cloneMaxPooling2DCC(e.getValue());
				} else if (e.getKey() == OpenCLSoftmax.class) 
				{
					OpenCLCore.getInstance().cloneSoftmaxFunction(e.getValue());
				} else if (e.getKey() == OpenCLStochasticPooling2D.class) 
				{
					OpenCLCore.getInstance().cloneStochasticPooling2DCC(e.getValue());
				} else if (e.getKey() == OpenCLAveragePooling2D.class) 
				{
					OpenCLCore.getInstance().cloneAveragePooling2DCC(e.getValue());
				} else if (e.getKey() == OpenCLMaxPooling2DBPCC.class) 
				{
					OpenCLCore.getInstance().cloneBackpropagationMaxPooling2DCC(e.getValue());
				} else if (e.getKey() == OpenCLAveragePooling2DBPCC.class) 
				{
					OpenCLCore.getInstance().cloneBackpropagationAveragePooling2DCC(e.getValue());
				} else if (e.getKey() == OpenCLLRNBPCC.class) 
				{
					OpenCLCore.getInstance().cloneBackPropagationLRN(e.getValue());
				} else if (e.getKey() == OpenCLConv2DBPWeightUpdates.class) 
				{
					OpenCLCore.getInstance().cloneBackpropagationConv2DWeightUpdates(e.getValue());
				} else if (e.getKey() == OpenCLFullyConnectedWeightUpdates.class) 
				{
					OpenCLCore.getInstance().cloneFullyConnectedWeightUpdates(e.getValue());
				} else if (e.getKey() == OpenCLMSE.class) 
				{
					OpenCLCore.getInstance().cloneMSEDerivative(e.getValue());
					//OpenCLCore.getInstance().cloneMSE(e.getValue());
				} else if (e.getKey() == OpenCLSoftmaxLoss.class) 
				{
					OpenCLCore.getInstance().cloneSoftmaxLoss(e.getValue());
					//OpenCLCore.getInstance().cloneNegativeLogProbability(e.getValue());
				} else if (e.getKey() == OpenCLSigmoid.class) 
				{
					OpenCLCore.getInstance().cloneSigmoid(e.getValue());
				} else if (e.getKey() == OpenCLTanh.class) 
				{
					OpenCLCore.getInstance().cloneTanh(e.getValue());
				} else if (e.getKey() == OpenCLReLU.class) 
				{
					OpenCLCore.getInstance().cloneReLU(e.getValue());
				} else if (e.getKey() == OpenCLSoftReLU.class) 
				{
					OpenCLCore.getInstance().cloneSoftReLU(e.getValue());
				} else if (e.getKey() == OpenCLReLUDerivative.class) 
				{
					OpenCLCore.getInstance().cloneReLUDerivative(e.getValue());
				} else if (e.getKey() == OpenCLTanhDerivative.class) 
				{
					OpenCLCore.getInstance().cloneTanhDerivative(e.getValue());
				} else if (e.getKey() == OpenCLSigmoidDerivative.class) 
				{
					OpenCLCore.getInstance().cloneSigmoidDerivative(e.getValue());
				} else if (e.getKey() == OpenCLSoftReLUDerivative.class) 
				{
					OpenCLCore.getInstance().cloneSoftReLUDerivative(e.getValue());
				} else if (e.getKey() == OpenCLMask.class) 
				{
					OpenCLCore.getInstance().cloneMask(e.getValue());
				} else if (e.getKey() == OpenCLNoise.class) 
				{
					OpenCLCore.getInstance().cloneNoise(e.getValue());
				} else if (e.getKey() == OpenCLNoiseMask.class) 
				{
					OpenCLCore.getInstance().cloneNoiseMask(e.getValue());
				} else if (e.getKey() == OpenCLFill.class) 
				{
					OpenCLCore.getInstance().cloneFill(e.getValue());
				} else if (e.getKey() == OpenCLClear.class) 
				{
					OpenCLCore.getInstance().cloneClear(e.getValue());
				} else 
				{
					throw new RuntimeException("Clone not found");
				}
			});

			// options mode
			devices.forEach(d -> OpenCLCore.getInstance().initDeviceID(d, deviceOptions.toString().trim().toCharArray(), true));
		}

		kernels.stream().filter(k -> km.get(k) == null).forEach(k -> km.put(k, k.createKernel()));
	}

	public char[] getJobs()
	{
		return jobs;
	}
}
