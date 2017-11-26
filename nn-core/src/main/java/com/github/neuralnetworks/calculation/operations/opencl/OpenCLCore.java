package com.github.neuralnetworks.calculation.operations.opencl;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.ClearValuesManager;


/**
 * Base class for OpenCL operations
 */
public class OpenCLCore
{
	private static OpenCLCore singleton = new OpenCLCore();

	public static final int EXBOCL_ERROR_JAVA_OK = 0; // no error
	public static final int EXBOCL_ERROR_JAVA_GENERAL = -1; // general error
	public static final int EXBOCL_ERROR_JAVA_MEMALLOCATE = -2; // could not allocate memory
	public static final int EXBOCL_ERROR_JAVA_OPENCLDEVICE = -3; // OpenCL platform/device not found
	public static final int EXBOCL_ERROR_JAVA_FINALIZE = -4; // could not finalize program/queue/context
	public static final int EXBOCL_ERROR_JAVA_RESOURCESNF = -5; // if the .dll/so cannot find the .cl-file with kernels

	private OCL ocl;
	private long openCLTime;
	private long kernelsRunTime;
	private List<Integer> availableDevices;
	private Set<Integer> initializedDevices;
	private Map<Integer, Integer> memoryUsage;
	private boolean devicesNumber;

	private OpenCLCore()
	{
		super();
		this.ocl = new OCL();
		this.availableDevices = new ArrayList<>();
		this.initializedDevices = new HashSet<>();
		this.memoryUsage = new HashMap<>();
	}

	public int getDevicesNumber()
	{
		devicesNumber = true;
		long start = System.nanoTime();
		int result = ocl.getDevicesNumber();
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException("getDevicesNumber failed with error " + result);
		}

		return result;
	}

	public int getDeviceID(int previous_deviceID)
	{
		return ocl.getDeviceID(previous_deviceID);
	}

	public int initDeviceID(int deviceID, char[] programOptions, boolean OptionsMode)
	{
		if (!devicesNumber)
		{
			getDevicesNumber();
		}

		long start = System.nanoTime();
		int result = ocl.initDeviceID(deviceID, programOptions == null ? new char[] {} : programOptions, OptionsMode);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException("Init device " + deviceID + " failed with error " + result);
		}

		initializedDevices.add(deviceID);

		return result;
	}

	public String getDeviceName(int deviceID)
	{
		long start = System.nanoTime();
		String result = ocl.getDeviceName(deviceID);
		openCLTime += System.nanoTime() - start;

		return result;
	}

	public String getPlatformName(int deviceID)
	{
		long start = System.nanoTime();
		String result = ocl.getPlatformName(deviceID);
		openCLTime += System.nanoTime() - start;

		return result;
	}

	public int initPRNG(int deviceID, int PRNGinstances, int Srand)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.initPRNG(deviceID, PRNGinstances, Srand);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int finalizeDeviceID(int deviceID)
	{
		initializedDevices.remove(deviceID);
		memoryUsage.remove(deviceID);
		OpenCLArrayReferenceManager.getInstance().clearDevice(deviceID);
		OpenCLKernelReferenceManager.getInstance().clearDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.finalizeDeviceID(deviceID);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int finalizeDeviceAll()
	{
		initializedDevices.clear();
		memoryUsage.clear();
		OpenCLArrayReferenceManager.getInstance().clear();
		OpenCLKernelReferenceManager.getInstance().clear();
		ClearValuesManager.getInstance().reset();
		devicesNumber = false;

		return ocl.finalizeDeviceAll();
	}

	public int prepareFloatArray(int deviceID, float[] input, int offset)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prepareFloatArray(deviceID, input, offset);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		if (!memoryUsage.containsKey(deviceID))
		{
			memoryUsage.put(deviceID, 0);
		}

		memoryUsage.put(deviceID, memoryUsage.get(deviceID) + input.length * 4);

		return result;
	}

	public int prepareFloatConstArray(int deviceID, float[] input, int offset)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prepareFloatConstArray(deviceID, input, offset);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		if (!memoryUsage.containsKey(deviceID))
		{
			memoryUsage.put(deviceID, 0);
		}

		memoryUsage.put(deviceID, memoryUsage.get(deviceID) + input.length * 4);

		return result;
	}

	public int prepareIntArray(int deviceID, int[] input, int offset)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prepareIntArray(deviceID, input, offset);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		if (!memoryUsage.containsKey(deviceID))
		{
			memoryUsage.put(deviceID, 0);
		}

		memoryUsage.put(deviceID, memoryUsage.get(deviceID) + input.length * 4);

		return result;
	}

	public int prepareIntConstArray(int deviceID, int[] input, int offset)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prepareIntConstArray(deviceID, input, offset);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		if (!memoryUsage.containsKey(deviceID))
		{
			memoryUsage.put(deviceID, 0);
		}

		memoryUsage.put(deviceID, memoryUsage.get(deviceID) + input.length * 4);

		return result;
	}

	public int getFloatBuf(int bufferID, float[] buffer)
	{
		long start = System.nanoTime();
		int result = ocl.getFloatBuf(bufferID, buffer);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int updateFloatBuf(int bufferID, float[] buffer)
	{
		long start = System.nanoTime();
		int result = ocl.updateFloatBuf(bufferID, buffer);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneFloatBuf(int bufferID, int deviceID)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.cloneFloatBuf(bufferID, deviceID);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int copyFloatBuf(int sourceID, int destinationID)
	{
		long start = System.nanoTime();
		int result = ocl.copyFloatBuf(sourceID, destinationID);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int weightedSum(int deviceID, int inputID, int weightsID, int outputID, int NDrange, int miniBatchSize,
			int inputStartPosition, int inputRowStep, int inputColumnStep, int outputStartPosition, int outputRowStep,
			int outputColumnStep, int weightStartPosition, int weightsSize, int weightsInitialStep, int weightsStep, boolean clear)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.weightedSum(deviceID, inputID, weightsID, outputID, NDrange, miniBatchSize, inputStartPosition, inputRowStep, inputColumnStep, outputStartPosition, outputRowStep, outputColumnStep, weightStartPosition, weightsSize, weightsInitialStep, weightsStep, clear);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Conv2DFF(int deviceID, int inputID, int weightsID, int outputID, int featureMapOffsetsID, int NDrange,
			int inputStartIndex, int inputFeatureMapRowsDistance, int inputFeatureMapColumnsDistance,
			int featureMapWeights, int outputColumns, int outputStartIndex, int outputFeatureMapLength,
			int outputFeatureMapsDistance, int outputFeatureMapColumnsDistance, int outputFeatureMapRowsDistance,
			int weightsStartIndex, int miniBatchSize, int inputMiniBatchDistance, int outputMiniBatchDistance,
			int numberFeatureMaps, int rowStride, int columnStride, int FilterRows, int FilterColumns, boolean clear)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Conv2DFF(deviceID, inputID, weightsID, outputID, featureMapOffsetsID, NDrange, inputStartIndex, inputFeatureMapRowsDistance, inputFeatureMapColumnsDistance, featureMapWeights, outputColumns, outputStartIndex, outputFeatureMapLength, outputFeatureMapsDistance, outputFeatureMapColumnsDistance, outputFeatureMapRowsDistance, weightsStartIndex, miniBatchSize, inputMiniBatchDistance, outputMiniBatchDistance, numberFeatureMaps, rowStride, columnStride, FilterRows, FilterColumns, clear);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BackpropagationConv2D2(int deviceID, int inputID, int weightsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapLength,
			int inputFeatureMapColumns, int inputFeatureMapRows, int inputFeatureMapColumnsDistance, int inputFeatureMapRowsDistance,
			int inputFeatureMapsDistance, int filterRows, int filterCols, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int outputFeatureMaps, int ioRowsOffset, int ioColumnsOffset, int rowStride,
			int columnStride, int weightsInputFiltersDistance, int weightsOutputFiltersDistance, int weightsStartIndex,
			int weightsRowsDistance, int weightsColumnsDistance, boolean clear)
	{
		checkInitDevice(deviceID);
		long start = System.nanoTime();
		int result = ocl.BackpropagationConv2D2(deviceID, inputID, weightsID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, inputFeatureMapLength, inputFeatureMapColumns, inputFeatureMapRows, inputFeatureMapColumnsDistance, inputFeatureMapRowsDistance, inputFeatureMapsDistance, filterRows, filterCols, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapRowsDistance, outputFeatureMapColumnsDistance, outputMiniBatchDistance, outputFeatureMaps, ioRowsOffset, ioColumnsOffset, rowStride, columnStride, weightsInputFiltersDistance, weightsOutputFiltersDistance, weightsStartIndex, weightsRowsDistance, weightsColumnsDistance, clear);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;

	}

	public int LRN(int deviceID, int inputID, int outputID, int cacheID, int NDrange, int inputFeatureMapsLength, int inputFeatureMaps, int inputStartIndex, int inputFeatureMapsDistance, int n, int miniBatchSize, int inputMiniBatchDistance, int outputStartIndex, float k, float a, float b)
	{
		checkInitDevice(deviceID);
		long start = System.nanoTime();
		int result = ocl.LRN(deviceID, inputID, outputID, cacheID, NDrange, inputFeatureMapsLength, inputFeatureMaps, inputStartIndex, inputFeatureMapsDistance, n, miniBatchSize, inputMiniBatchDistance, outputStartIndex, k, a, b);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BackPropagationLRN(int deviceID, int inputID, int casheID, int activationsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapsLength,
			int inputFeatureMaps, int activationsStartIndex, int activationsFeatureMapsDistance, int activationsFeatureMapsLength,
			int activationsMiniBatchDistance, int outputStartIndex, int inputFeatureMapsDistance, int n, float a,
			float b)
	{
		checkInitDevice(deviceID);
		long start = System.nanoTime();
		int result = ocl.BackPropagationLRN(deviceID, inputID, casheID, activationsID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, inputFeatureMapsLength, inputFeatureMaps, activationsStartIndex, activationsFeatureMapsDistance, activationsFeatureMapsLength, activationsMiniBatchDistance, outputStartIndex, inputFeatureMapsDistance, n, a, b);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int MaxoutFunction(int deviceID, int inputID, int weightsID, int maxoutWinnersID, int outputID, int NDrange, int miniBatchSize, int inputStartPosition, int inputColumnStep,
			int weightStartPosition, int weightsSize, int inputRowStep, int weightsInitialStep, int weightsStep, int winnersStartPosition, int outputStartPosition, int outputRowStep, int outputColumnStep)
	{
		checkInitDevice(deviceID);
		long start = System.nanoTime();
		int result = ocl.MaxoutFunction(deviceID, inputID, weightsID, maxoutWinnersID, outputID, NDrange, miniBatchSize, inputStartPosition, inputColumnStep, weightStartPosition, weightsSize,
				inputRowStep, weightsInitialStep, weightsStep, winnersStartPosition, outputStartPosition, outputRowStep, outputColumnStep);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BackPropagationMaxout(int deviceID, int inputID, int weightsID, int weightUpdatesID, int ffActivationID, int maxoutWinnersID, int outputID, int NDrange, int miniBatchSize,
			int weightStartPosition, int weightsInitialStep, int weightsStep, int winnersStartPosition, int outputStartPosition, int outputRowStep, int outputColumnStep, int activationStartPosition,
			int activationRowStep, int activationColumnStep, float learningRate, float momentum, float l1weightDecay, float l2weightDecay)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.BackPropagationMaxout(deviceID, inputID, weightsID, weightUpdatesID, ffActivationID, maxoutWinnersID, outputID, NDrange, miniBatchSize, weightStartPosition, weightsInitialStep,
				weightsStep, winnersStartPosition, outputStartPosition, outputRowStep, outputColumnStep, activationStartPosition, activationRowStep, activationColumnStep, learningRate, momentum,
				l1weightDecay, l2weightDecay);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BackpropagationConv2DWeightUpdates(int deviceID, int activationID, int weightsID, int weightsUpdatesID, int outputID, int NDrange,
			int miniBatchSize, int weightsRowsDistance, int weightsColumnsDistance, int weightsStartIndex,
			int weightsUpdatesOutputFiltersDistance, int weightsUpdatesInputFiltersDistance, int weightsInputFiltersDistance, int weightsOutputFiltersDistance,
			int weightsUpdatesRowsDistance, int activationStartIndex, int activationFeatureMapsDistance, int activationFeatureMapRowsDistance,
			int activationFeatureMapColumnsDistance, int activationMiniBatchDistance, int outputFeatureMapRows, int outputFeatureMapColumns,
			int outputMiniBatchDistance, int outputStartIndex, int outputFeatureMapsDistance, int outputFeatureMapRowsDistance,
			int outputFeatureMapColumnsDistance, int rowStride, int columnStride,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.BackpropagationConv2DWeightUpdates(deviceID, activationID, weightsID, weightsUpdatesID, outputID, NDrange, miniBatchSize, weightsRowsDistance, weightsColumnsDistance,
				weightsStartIndex, weightsUpdatesOutputFiltersDistance, weightsUpdatesInputFiltersDistance, weightsInputFiltersDistance, weightsOutputFiltersDistance, weightsUpdatesRowsDistance,
				activationStartIndex, activationFeatureMapsDistance, activationFeatureMapRowsDistance, activationFeatureMapColumnsDistance, activationMiniBatchDistance, outputFeatureMapRows,
				outputFeatureMapColumns, outputMiniBatchDistance, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapRowsDistance, outputFeatureMapColumnsDistance, rowStride, columnStride,
				learningRate, momentum, l1weightDecay, l2weightDecay);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int FullyConnectedWeightUpdates(int deviceID, int inputID, int weightsID, int weightUpdatesID, int ffActivationID, int NDrange,
			int miniBatchSize, int inputStartPosition, int inputColumnStep, int inputRowStep,
			int activationStartPosition, int activationColumnStep, int activationRowStep, int weightStartPosition,
			int weightsColumns, int weightsRowsDistance, int weightsColumnsDistance,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.FullyConnectedWeightUpdates(deviceID, inputID, weightsID, weightUpdatesID, ffActivationID, NDrange, miniBatchSize, inputStartPosition, inputColumnStep, inputRowStep,
				activationStartPosition, activationColumnStep, activationRowStep, weightStartPosition, weightsColumns, weightsRowsDistance, weightsColumnsDistance, learningRate, momentum, l1weightDecay,
				l2weightDecay);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int MaxPooling2DCC(int deviceID, int inputID, int featureMapOffsetsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapLength, int outputFeatureMapColumns,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int ioRowsOffset, int ioColumnsOffset, int rowStride, int columnStride, int regionLength)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.MaxPooling2DCC(deviceID, inputID, featureMapOffsetsID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, inputFeatureMapColumnsDistance, inputFeatureMapRowsDistance, inputFeatureMapsDistance, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapLength, outputFeatureMapColumns, outputFeatureMapRowsDistance, outputFeatureMapColumnsDistance, outputMiniBatchDistance, ioRowsOffset, ioColumnsOffset, rowStride, columnStride, regionLength);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int StochasticPooling2DCC(int deviceID, int inputID, int featureMapOffsetsID, int outputID, int NDrange, int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance,
			int inputFeatureMapColumnsDistance, int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int outputStartIndex, int outputFeatureMapsDistance, int outputFeatureMapLength,
			int outputFeatureMapColumns, int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance, int ioRowsOffset, int ioColumnsOffset, int rowStride,
			int columnStride, int regionLength)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.StochasticPooling2DCC(deviceID, inputID, featureMapOffsetsID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, inputFeatureMapColumnsDistance,
				inputFeatureMapRowsDistance, inputFeatureMapsDistance, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapLength, outputFeatureMapColumns, outputFeatureMapRowsDistance,
				outputFeatureMapColumnsDistance, outputMiniBatchDistance, ioRowsOffset, ioColumnsOffset, rowStride, columnStride, regionLength);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int AveragePooling2DCC(int deviceID, int inputID, int featureMapOffsetsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapLength, int outputFeatureMapColumns,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int ioRowsOffset, int ioColumnsOffset, int rowStride, int columnStride, int regionLength)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.AveragePooling2DCC(deviceID, inputID, featureMapOffsetsID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, inputFeatureMapColumnsDistance, inputFeatureMapRowsDistance, inputFeatureMapsDistance, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapLength, outputFeatureMapColumns, outputFeatureMapRowsDistance, outputFeatureMapColumnsDistance, outputMiniBatchDistance, ioRowsOffset, ioColumnsOffset, rowStride, columnStride, regionLength);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BackpropagationMaxPooling2DCC(int deviceID, int inputID, int featureMapOffsetsID,
			int ffActivationID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int inputFeatureMapLength,
			int inputFeatureMapColumns, int inputFeatureMapRows, int subsamplingRows, int subsamplingCols,
			int outputMiniBatchDistance, int outputStartIndex, int outputFeatureMapsDistance,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int ioColumnsOffset,
			int ioRowsOffset, int rowStride, int columnStride, int regionLength)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.BackpropagationMaxPooling2DCC(deviceID, inputID, featureMapOffsetsID, ffActivationID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, inputFeatureMapColumnsDistance, inputFeatureMapRowsDistance, inputFeatureMapsDistance, inputFeatureMapLength, inputFeatureMapColumns, inputFeatureMapRows, subsamplingRows, subsamplingCols, outputMiniBatchDistance, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapRowsDistance, outputFeatureMapColumnsDistance, ioColumnsOffset, ioRowsOffset, rowStride, columnStride, regionLength);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BackpropagationAveragePooling2DCC(int deviceID, int inputID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int regionLength,
			int inputFeatureMapLength, int inputFeatureMapColumns,
			int inputFeatureMapRows, int subsamplingRows, int subsamplingCols, int outputMiniBatchDistance,
			int outputStartIndex, int outputFeatureMapsDistance, int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance,
			int ioColumnsOffset, int ioRowsOffset, int rowStride, int columnStride)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.BackpropagationAveragePooling2DCC(deviceID, inputID, outputID, NDrange, miniBatchSize, inputStartIndex, inputMiniBatchDistance, regionLength, inputFeatureMapLength, inputFeatureMapColumns, inputFeatureMapRows, subsamplingRows, subsamplingCols, outputMiniBatchDistance, outputStartIndex, outputFeatureMapsDistance, outputFeatureMapRowsDistance, outputFeatureMapColumnsDistance, ioColumnsOffset, ioRowsOffset, rowStride, columnStride);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int SoftmaxFunction(int deviceID, int inputID, int NDrange, int startIndex, int columns, int nextRowStep, int nextColumnStep)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.SoftmaxFunction(deviceID, inputID, NDrange, startIndex, columns, nextRowStep, nextColumnStep);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int MSEDerivative(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int activationStartPosition, int activationRowStep, int activationColumnStep, int targetStartPosition,
			int targetRowStep, int targetColumnStep, int resultStartPosition, int resultRowStep,
			int resultColumnStep)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.MSEDerivative(deviceID, activationID, targetID, resultID, NDrange, activationStartPosition, activationRowStep, activationColumnStep, targetStartPosition, targetRowStep,
				targetColumnStep, resultStartPosition, resultRowStep, resultColumnStep);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;

	}

	public int SoftmaxLoss(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int activationStartPosition, int activationRowStep, int activationColumnStep, int targetStartPosition,
			int targetRowStep, int targetColumnStep, int resultStartPosition, int resultRowStep,
			int resultColumnStep, int reverse)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.SoftmaxLoss(deviceID, activationID, targetID, resultID, NDrange, activationStartPosition, activationRowStep, activationColumnStep, targetStartPosition, targetRowStep,
				targetColumnStep, resultStartPosition, resultRowStep, resultColumnStep, reverse);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int NegativeLogProbability(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int miniBatchSize, int activationStartPosition, int activationRowStep, int activationColumnStep,
			int targetStartPosition, int targetRowStep, int targetColumnStep, int resultStartPosition,
			int resultRowStep, int resultColumnStep)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.NegativeLogProbability(deviceID, activationID, targetID, resultID, NDrange, miniBatchSize, activationStartPosition, activationRowStep, activationColumnStep, targetStartPosition,
				targetRowStep, targetColumnStep, resultStartPosition, resultRowStep, resultColumnStep);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int MSE(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int activationStartPosition, int activationRowStep, int activationColumnStep, int targetStartPosition,
			int targetRowStep, int targetColumnStep, int resultStartPosition, int resultRowStep,
			int resultColumnStep)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.MSE(deviceID, activationID, targetID, resultID, NDrange, activationStartPosition, activationRowStep, activationColumnStep, targetStartPosition, targetRowStep, targetColumnStep,
				resultStartPosition, resultRowStep, resultColumnStep);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Sigmoid(int deviceID, int ioarrayID, int NDrange, int startIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Sigmoid(deviceID, ioarrayID, NDrange, startIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Tanh(int deviceID, int ioarrayID, int NDrange, int startIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Tanh(deviceID, ioarrayID, NDrange, startIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int ReLU(int deviceID, int ioarrayID, int NDrange, int startIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.ReLU(deviceID, ioarrayID, NDrange, startIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int SoftReLU(int deviceID, int ioarrayID, int NDrange, int startIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.SoftReLU(deviceID, ioarrayID, NDrange, startIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int ReLUDerivative(int deviceID, int resultsID, int activationsID, int NDrange, int activationsStartIndex, int resultStartIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.ReLUDerivative(deviceID, resultsID, activationsID, NDrange, activationsStartIndex, resultStartIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int TanhDerivative(int deviceID, int resultsID, int activationsID, int NDrange, int activationsStartIndex, int resultStartIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.TanhDerivative(deviceID, resultsID, activationsID, NDrange, activationsStartIndex, resultStartIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int SigmoidDerivative(int deviceID, int resultsID, int activationsID, int NDrange, int activationsStartIndex, int resultStartIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.SigmoidDerivative(deviceID, resultsID, activationsID, NDrange, activationsStartIndex, resultStartIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int SoftReLUDerivative(int deviceID, int resultsID, int activationsID, int NDrange, int activationsStartIndex, int resultStartIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.SoftReLUDerivative(deviceID, resultsID, activationsID, NDrange, activationsStartIndex, resultStartIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int prng(int deviceID, int outputID, int NDrange)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prng(deviceID, outputID, NDrange);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int prngRestart(int deviceID)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prngRestart(deviceID);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int prngGaussian(int deviceID, int outputID, int NDrange)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.prngGaussian(deviceID, outputID, NDrange);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int BernoulliKernel(int deviceID, int outputID, int NDrange)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.BernoulliKernel(deviceID, outputID, NDrange);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Noise(int deviceID, int ioarrayID, int NDrange, int activationsStartIndex, float corruptionLevel, float corruptedValue)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Noise(deviceID, ioarrayID, NDrange, activationsStartIndex, corruptionLevel, corruptedValue);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int NoiseMask(int deviceID, int ioarrayID, int maskArrayID, int NDrange,
			int activationsStartIndex, int maskStartIndex, float corruptionLevel, float corruptedValue)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.NoiseMask(deviceID, ioarrayID, maskArrayID, NDrange, activationsStartIndex, maskStartIndex, corruptionLevel, corruptedValue);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Mask(int deviceID, int inputOutputID, int maskArrayID, int NDRange,
			int inputStartIndex, int maskStartIndex)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Mask(deviceID, inputOutputID, maskArrayID, NDRange, inputStartIndex, maskStartIndex);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Fill(int deviceID, int arrayID, float value)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Fill(deviceID, arrayID, value);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int Clear(int deviceID, int arrayID)
	{
		checkInitDevice(deviceID);

		long start = System.nanoTime();
		int result = ocl.Clear(deviceID, arrayID);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int kernelRun(int kernelID)
	{
		long start = System.nanoTime();
		int result = ocl.kernelRun(kernelID);
		kernelsRunTime += System.nanoTime() - start;
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int kernelRunAsync(int kernelID)
	{
		long start = System.nanoTime();
		int result = ocl.kernelRunAsync(kernelID);
		kernelsRunTime += System.nanoTime() - start;
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int kernelRunJob(char[] job)
	{
		if (job == null || job.length == 0)
		{
			throw new RuntimeException("empty jobs string");
		}

		long start = System.nanoTime();
		int result = ocl.kernelRunJob(job);
		kernelsRunTime += System.nanoTime() - start;
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int kernelState(int kernelID)
	{
		return ocl.kernelState(kernelID);
	}

	public int setProfilingMode()
	{
		long start = System.nanoTime();
		int result = ocl.setProfilingMode();
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int kernelPrintProfilingInfo(int kernelID)
	{
		long start = System.nanoTime();
		int result = ocl.kernelPrintProfilingInfo(kernelID);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int pathSetToCL(char[] path)
	{
		long start = System.nanoTime();
		int result = ocl.pathSetToCL(path);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int checkLibraryLoad()
	{
		long start = System.nanoTime();
		int result = ocl.checkLibraryLoad();
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int logMessage(char[] message)
	{
		long start = System.nanoTime();
		int result = ocl.logMessage(message);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}
	
	public int restartLibrary()
	{
		initializedDevices.clear();
		memoryUsage.clear();
		OpenCLArrayReferenceManager.getInstance().clear();
		OpenCLKernelReferenceManager.getInstance().clear();
		ClearValuesManager.getInstance().reset();
		devicesNumber = false;
		openCLTime = kernelsRunTime = 0;

		long start = System.nanoTime();
		int result = ocl.restartLibrary();
		openCLTime += System.nanoTime() - start;
		
		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneWeightedSum(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneWeightedSum(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneConv2DFF(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneConv2DFF(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneLRN(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneLRN(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneMaxoutFunction(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneMaxoutFunction(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneBackPropagationMaxout(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneBackPropagationMaxout(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneBackpropagationConv2D2(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneBackpropagationConv2D2(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneMaxPooling2DCC(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneMaxPooling2DCC(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneSoftmaxFunction(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneSoftmaxFunction(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneStochasticPooling2DCC(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneStochasticPooling2DCC(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneAveragePooling2DCC(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneAveragePooling2DCC(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneBackpropagationMaxPooling2DCC(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneBackpropagationMaxPooling2DCC(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneBackpropagationAveragePooling2DCC(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneBackpropagationAveragePooling2DCC(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneBackPropagationLRN(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneBackPropagationLRN(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneBackpropagationConv2DWeightUpdates(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneBackpropagationConv2DWeightUpdates(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneFullyConnectedWeightUpdates(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneFullyConnectedWeightUpdates(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneMSEDerivative(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneMSEDerivative(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneSoftmaxLoss(int N) 
	{
		long start = System.nanoTime();
		int result = ocl.cloneSoftmaxLoss(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneNegativeLogProbability(int N) 
	{
		long start = System.nanoTime();
		int result = ocl.cloneNegativeLogProbability(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneMSE(int N) 
	{
		long start = System.nanoTime();
		int result = ocl.cloneMSE(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneSigmoid(int N) 
	{
		long start = System.nanoTime();
		int result = ocl.cloneSigmoid(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneTanh(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneTanh(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneReLU(int N) {
		long start = System.nanoTime();
		int result = ocl.cloneReLU(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneSoftReLU(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneSoftReLU(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneReLUDerivative(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneReLUDerivative(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneTanhDerivative(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneTanhDerivative(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneSigmoidDerivative(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneSigmoidDerivative(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneSoftReLUDerivative(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneSoftReLUDerivative(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneMask(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneMask(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneNoise(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneNoise(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneNoiseMask(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneNoiseMask(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}

	public int cloneFill(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneFill(N);
		openCLTime += System.nanoTime() - start;

		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}

		return result;
	}
	
	public int cloneClear(int N)
	{
		long start = System.nanoTime();
		int result = ocl.cloneClear(N);
		openCLTime += System.nanoTime() - start;
		
		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}
		
		return result;
	}

	public int checkSetPrecision(float precision) 
	{
		long start = System.nanoTime();
		int result = ocl.checkSetPrecision(precision);
		openCLTime += System.nanoTime() - start;
		
		if (result < EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}
		
		return result;
	}

	public int checkIntBuf(int bufferID, int[] data)
	{
		long start = System.nanoTime();
		int result = ocl.checkIntBuf(bufferID, data);
		openCLTime += System.nanoTime() - start;
		
		if (result != EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}
		
		return result;
	}

	public int checkFloatBuf(int bufferID, float[] data) 
	{
		long start = System.nanoTime();
		int result = ocl.checkFloatBuf(bufferID, data);
		openCLTime += System.nanoTime() - start;
		
		if (result != EXBOCL_ERROR_JAVA_OK)
		{
			throw new RuntimeException(Thread.currentThread().getStackTrace()[1].getMethodName() + " error: " + result);
		}
		
		return result;
	}

	public Set<Integer> getInitializedDevices()
	{
		return initializedDevices;
	}

	public List<Integer> getAvailableDevices()
	{
		if (availableDevices.size() == 0)
		{
			int devices = getDevicesNumber();
			for (int i = 0; i < devices; i++)
			{
				availableDevices.add(getDeviceID(i));
			}
		}

		return availableDevices;
	}

	public long getOpenCLTime()
	{
		return openCLTime / 1000;
	}
	
	public long getKernelsRunTime()
	{
		return kernelsRunTime / 1000;
	}

	public Map<Integer, Integer> getMemoryUsage()
	{
		return memoryUsage;
	}

	public static OpenCLCore getInstance()
	{
		return singleton;
	}

	public static String getKernelOptionsString(Kernel kernel)
	{
		StringBuilder result = new StringBuilder();
		Map<String, Object> kernelOptions = getKernelOptions(kernel);
		kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append("=").append(e.getValue()));

		return result.toString();
	}

	public static Map<String, Object> getKernelOptions(Kernel kernel)
	{
		Map<String, Object> result = new HashMap<>();

		Class<?> current = kernel.getClass();
		while (current != null && current != Kernel.class)
		{
			for (Field f : current.getDeclaredFields())
			{
				Class<?> type = f.getType();
				if (type.isPrimitive() && (type.equals(int.class) || type.equals(float.class) || type.equals(double.class)))
				{
					boolean isAccessible = f.isAccessible();
					f.setAccessible(true);

					try
					{
						result.put(current.getSimpleName() + "_" + f.getName(), f.get(kernel));
					} catch (IllegalArgumentException | IllegalAccessException e)
					{
						e.printStackTrace();
					}

					f.setAccessible(isAccessible);
				}
			}

			current = current.getSuperclass();
		}

		return result;
	}
	
	public static String getKernelOptionsString(Kernel kernel, Map<String, String> fieldsMap, int order)
	{
		StringBuilder result = new StringBuilder();
		Map<String, Object> kernelOptions = getKernelOptions(kernel, fieldsMap);
		kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

		return result.toString();
	}

	public static Map<String, Object> getKernelOptions(Kernel kernel, Map<String, String> fieldsMap)
	{ 
		Map<String, Object> result = new HashMap<>();
		
		Class<?> current = kernel.getClass();
		while (current != null && current != Kernel.class)
		{
			for (Field f : current.getDeclaredFields())
			{
				Class<?> type = f.getType();
				if (type.isPrimitive() && (type.equals(int.class) || type.equals(float.class) || type.equals(double.class)))
				{
					boolean isAccessible = f.isAccessible();
					f.setAccessible(true);

					try
					{
						if (fieldsMap.containsKey(f.getName()))
						{
							result.put(fieldsMap.get(f.getName()), f.get(kernel) + (f.getType().getName().equals("float") ? "f" : ""));
						}
					} catch (IllegalArgumentException | IllegalAccessException e)
					{
						e.printStackTrace();
					}
					
					f.setAccessible(isAccessible);
				}
			}
			
			current = current.getSuperclass();
		}

		return result;
	}

	private void checkInitDevice(int deviceID)
	{
		if (!initializedDevices.contains(deviceID))
		{
			initDeviceID(deviceID, null, false);
		}
	}
}
