package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.SystemUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class OCL
{
	private static final Logger LOGGER = LoggerFactory.getLogger(OCL.class);

	private static final String CL_FILES_ENVIRONMENT_VARIABLE = "EXBJCL";
	private static final String LIBRARY_ENVIRONMENT_VARIABLE = "EXBJCLDLL";

	private static final String mWindowsLibrary = "ExBJCL.dll";
	private static final String mLinuxLibrary = "ExBJCL.so";
	private static final String[] exboclkernels = new String[] { "exboclkernels.cl", "exboclkernels_core.cl", "exboclkernels_templates.cl" };
	private static final String mNativePath = "native";
	private static final String mNativePathIntel = mNativePath + "/" + "intel";
	private static final String mCLPath = "cl";
	private static final String mNativePathAMD = mNativePath + "/" + "amd";
	private static final String mNativePathNvidia = mNativePath + "/" + "nvidia";

	private static String clFilesPath = null;

	{
		if (clFilesPath == null)
		{
			if (System.getenv(LIBRARY_ENVIRONMENT_VARIABLE) == null || System.getenv(LIBRARY_ENVIRONMENT_VARIABLE).isEmpty() || System.getenv(CL_FILES_ENVIRONMENT_VARIABLE) == null
					|| System.getenv(CL_FILES_ENVIRONMENT_VARIABLE).isEmpty())
			{
				loadNativeCodeFromJar();
			} else
			{
				LOGGER.warn("I will load now the the experimental version from the variable " + System.getenv(LIBRARY_ENVIRONMENT_VARIABLE) + " and not the libraries from the jar!");
				loadNativeCodeWithGlobalEnvironmentVariables();
			}
			checkError(pathSetToCL(clFilesPath.toCharArray()));
		} else
		{
			LOGGER.warn("Native code was already loaded");
		}
	}

	/**
	 * extract the lib, co and both cl files into the temporary directory (override old one), load the lib or so depending on the system and set the local path to the cl files
	 * 
	 * @throws IllegalStateException
	 *           will be thrown if the clFilesPath isn't null = the native code is already loaded
	 */
	public static void loadNativeCodeFromJar()
	{
		loadNativeCodeFromJar(ARCHITECTURE.NVIDIA);
	}

	public enum ARCHITECTURE
	{
		INTEL,
		AMD,
		NVIDIA;
	}

	public static void loadNativeCodeFromJar(final ARCHITECTURE architecture)
	{

		if (architecture == null)
		{
			throw new IllegalArgumentException("Architekture must not be null");
		}

		if (clFilesPath != null)
		{
			throw new IllegalStateException("The native code is already loaded!");
		}

		File dllName;
		File soName;

		// extract libraries
		File tempDir;
		try
		{
			tempDir = Files.createTempDirectory("opencl_").toFile();
		} catch (IOException e)
		{
			throw new IllegalStateException("Couldn't write temporary file");
		}
		clFilesPath = tempDir.getAbsolutePath();

		dllName = new File(tempDir, mWindowsLibrary);
		soName = new File(tempDir, mLinuxLibrary);

		dllName.deleteOnExit();
		soName.deleteOnExit();

		// extract cl files
		for (String kernel : OCL.exboclkernels)
		{
			File kernelFile = new File(tempDir, kernel);
			kernelFile.deleteOnExit();
			CopyLibrary(mCLPath + "/" + kernel, kernelFile.getAbsolutePath());
		}

		// copy the libraries from the jar to the temp directory
		if (architecture.equals(ARCHITECTURE.INTEL))
		{
			CopyLibrary(mNativePathIntel + "/" + mWindowsLibrary, dllName.getAbsolutePath());
			CopyLibrary(mNativePathIntel + "/" + mLinuxLibrary, soName.getAbsolutePath());
		} else if (architecture.equals(ARCHITECTURE.AMD))
		{
			CopyLibrary(mNativePathAMD + "/" + mWindowsLibrary, dllName.getAbsolutePath());
			CopyLibrary(mNativePathAMD + "/" + mLinuxLibrary, soName.getAbsolutePath());
		} else if (architecture.equals(ARCHITECTURE.NVIDIA))
		{
			CopyLibrary(mNativePathNvidia + "/" + mWindowsLibrary, dllName.getAbsolutePath());
			CopyLibrary(mNativePathNvidia + "/" + mLinuxLibrary, soName.getAbsolutePath());
		} else
		{
			throw new IllegalArgumentException("Unexpected architecture request");
		}

        System.out.println("load TC lib");


        String osName = SystemUtils.OS_NAME;
		if (osName.startsWith("Windows"))
		{
			System.load(dllName.getAbsolutePath());
		} else if (osName.startsWith("Linux"))
		{
			System.load(soName.getAbsolutePath());
		} else
		{
			throw new IllegalStateException("Not supported os " + osName);
		}

	}


	private static void CopyLibrary(String aResource, String aTarget)
	{
		InputStream is = null;
		OutputStream os = null;
		try
		{
			is = Thread.currentThread().getContextClassLoader().getResourceAsStream(aResource);
			if (is == null)
			{
				throw new IllegalArgumentException("Could not found resource " + aResource + " inresource path");
			}
			os = new FileOutputStream(aTarget);

			IOUtils.copy(is, os);
		} catch (IOException ex)
		{
			LOGGER.error(ex.getMessage(), ex);
		} finally
		{
			IOUtils.closeQuietly(is);
			IOUtils.closeQuietly(os);
		}
	}

	/**
	 * search the so and dll within the paths that are in the global envoronment variable EXBJCL, load the lib or so (whatever was found first)
	 * and set the local path to the cl files which should be in the same directory
	 * 
	 * @throws IllegalStateException
	 *           will be thrown if the clFilesPath isn't null = the native code is already loaded
	 */
	public static void loadNativeCodeWithGlobalEnvironmentVariables() throws IllegalStateException
	{
        System.out.println("load local lib");

		if (clFilesPath != null)
		{
			throw new IllegalStateException("The native code is already loaded!!!!! (" + clFilesPath + ")");
		}

		// check if cl files and library exists
		File clDirectory = new File(System.getenv(CL_FILES_ENVIRONMENT_VARIABLE));

		if (!clDirectory.exists())
		{
			LOGGER.warn("Load the native code from jar because the EXBJCL directory ("
					+ clDirectory.toString() + ") doesn't exist!");
			loadNativeCodeFromJar();
			return;
		}

		File[] clFiles = clDirectory.listFiles(new FileFilter()
		{
			@Override
			public boolean accept(File pathname)
			{
				return pathname.getName().endsWith(".cl");
			}
		});

		if (clFiles == null || clFiles.length == 0)
		{
			LOGGER.warn("Load the native code from jar because the EXBJCL directory ("
					+ clDirectory.toString() + ") doesn't contain any .cl files!");
			loadNativeCodeFromJar();
			return;
		}

		Path nativeCodePath = Paths.get(System.getenv(LIBRARY_ENVIRONMENT_VARIABLE));

		// check if library exists
		if (!nativeCodePath.toFile().exists() || nativeCodePath.toFile().isDirectory())
		{
			LOGGER.warn("Load the native code from jar because the EXBJCLDLL library ("
					+ nativeCodePath.toString() + ") doesn't exist or isn't a file!");
			loadNativeCodeFromJar();
			return;
		}

		// load lib and set path
		clFilesPath = System.getenv(CL_FILES_ENVIRONMENT_VARIABLE);
		System.load(nativeCodePath.toString());
	}

	public OCL()
	{

	}

	public static void checkError(int result)
	{
		if (result != 0)
		{
			LOGGER.error("[ERROR!] " + result);
		}
	}

	public native int getDevicesNumber();

	public native int getDeviceID(int previous_deviceID);

	public native String getDeviceName(int deviceID);

	public native String getPlatformName(int deviceID);

	public native int initDeviceID(int deviceID, char[] programOptions, boolean OptionsMode);

	public native int initPRNG(int deviceID, int PRNGinstances, int Srand);

	// Srand - if =0, then PRNG is initialized with timer, else with Srand value
	// PRNGinstances - max number of generators
	public native int finalizeDeviceID(int deviceID);

	public native int finalizeDeviceAll();

	public native int setProfilingMode();

	// public native int setOptionsMode( char[] programOptions );
	public native int getOptimizationState();

	public native int prepareFloatArray(int deviceID, float[] input, int offset);

	public native int prepareFloatConstArray(int deviceID, float[] input, int offset);

	public native int prepareIntArray(int deviceID, int[] input, int offset);

	public native int prepareIntConstArray(int deviceID, int[] input, int offset);

	public native int prepareBuf(int inputID, int[] dimensions);

	public native int getFloatBuf(int bufferID, float[] buffer);

	public native int updateFloatBuf(int bufferID, float[] buffer);

	public native int cloneFloatBuf(int bufferID, int deviceID);

	public native int copyFloatBuf(int sourceID, int destinationID);

	public native int checkSetPrecision(float precision);

	public native int checkIntBuf(int bufferID, int[] data);

	public native int checkFloatBuf(int bufferID, float[] data);

	public native int saveBuf(int bufferID);

	public native int weightedSum(int deviceID, int inputID, int weightsID, int outputID, int NDrange, int miniBatchSize,
			int inputStartPosition, int inputRowStep, int inputColumnStep, int outputStartPosition, int outputRowStep,
			int outputColumnStep, int weightStartPosition, int weightsSize, int weightsInitialStep, int weightsStep, boolean clear);

	public native int Conv2DFF(int deviceID, int inputID, int weightsID, int outputID, int featureMapOffsetsID, int NDrange,
			int inputStartIndex, int inputFeatureMapRowsDistance, int inputFeatureMapColumnsDistance,
			int featureMapWeights, int outputColumns, int outputStartIndex, int outputFeatureMapLength,
			int outputFeatureMapsDistance, int outputFeatureMapColumnsDistance, int outputFeatureMapRowsDistance,
			int weightsStartIndex, int miniBatchSize, int inputMiniBatchDistance, int outputMiniBatchDistance,
			int numberFeatureMaps, int rowStride, int columnStride, int FilterRows, int FilterColumns, boolean clear);

	public native int LRN(int deviceID, int inputID, int outputID, int cacheID, int NDrange,
			int inputFeatureMapsLength, int inputFeatureMaps, int inputStartIndex, int inputFeatureMapsDistance,
			int n, int miniBatchSize, int inputMiniBatchDistance, int outputStartIndex, float k, float a, float b);

	public native int MaxoutFunction(int deviceID, int inputID, int weightsID, int maxoutWinnersID, int outputID, int NDrange,
			int miniBatchSize, int inputStartPosition, int inputColumnStep, int weightStartPosition,
			int weightsSize, int inputRowStep, int weightsInitialStep, int weightsStep,
			int winnersStartPosition, int outputStartPosition, int outputRowStep, int outputColumnStep);

	public native int BackPropagationMaxout(int deviceID, int inputID, int weightsID, int weightUpdatesID,
			int ffActivationID, int maxoutWinnersID, int outputID, int NDrange,
			int miniBatchSize, int weightStartPosition, int weightsInitialStep, int weightsStep,
			int winnersStartPosition, int outputStartPosition, int outputRowStep, int outputColumnStep,
			int activationStartPosition, int activationRowStep, int activationColumnStep,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay_2);

	public native int BackpropagationConv2D2(int deviceID, int inputID, int weightsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapLength,
			int inputFeatureMapColumns, int inputFeatureMapRows, int inputFeatureMapColumnsDistance, int inputFeatureMapRowsDistance,
			int inputFeatureMapsDistance, int filterRows, int filterCols, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int outputFeatureMaps, int ioRowsOffset, int ioColumnsOffset, int rowStride,
			int columnStride, int weightsInputFiltersDistance, int weightsOutputFiltersDistance, int weightsStartIndex,
			int weightsRowsDistance, int weightsColumnsDistance, boolean clear);

	public native int MaxPooling2DCC(int deviceID, int inputID, int featureMapOffsetsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapLength, int outputFeatureMapColumns,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int ioRowsOffset, int ioColumnsOffset, int rowStride, int columnStride, int regionLength);

	public native int SoftmaxFunction(int deviceID, int valuesID, int NDrange,
			int startIndex, int columns, int nextRowStep, int nextColumnStep);

	public native int StochasticPooling2DCC(int deviceID, int inputID, int featureMapOffsetsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapLength, int outputFeatureMapColumns,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int ioRowsOffset, int ioColumnsOffset, int rowStride, int columnStride, int regionLength);

	public native int AveragePooling2DCC(int deviceID, int inputID, int featureMapOffsetsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int outputStartIndex,
			int outputFeatureMapsDistance, int outputFeatureMapLength, int outputFeatureMapColumns,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int outputMiniBatchDistance,
			int ioRowsOffset, int ioColumnsOffset, int rowStride, int columnStride, int regionLength);

	public native int BackpropagationMaxPooling2DCC(int deviceID, int inputID, int featureMapOffsetsID,
			int ffActivationID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapColumnsDistance,
			int inputFeatureMapRowsDistance, int inputFeatureMapsDistance, int inputFeatureMapLength,
			int inputFeatureMapColumns, int inputFeatureMapRows, int subsamplingRows, int subsamplingCols,
			int outputMiniBatchDistance, int outputStartIndex, int outputFeatureMapsDistance,
			int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance, int ioColumnsOffset,
			int ioRowsOffset, int rowStride, int columnStride, int regionLength);

	public native int BackpropagationAveragePooling2DCC(int deviceID, int inputID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int regionLength,
			int inputFeatureMapLength, int inputFeatureMapColumns,
			int inputFeatureMapRows, int subsamplingRows, int subsamplingCols, int outputMiniBatchDistance,
			int outputStartIndex, int outputFeatureMapsDistance, int outputFeatureMapRowsDistance, int outputFeatureMapColumnsDistance,
			int ioColumnsOffset, int ioRowsOffset, int rowStride, int columnStride);

	public native int BackPropagationLRN(int deviceID, int inputID, int casheID, int activationsID, int outputID, int NDrange,
			int miniBatchSize, int inputStartIndex, int inputMiniBatchDistance, int inputFeatureMapsLength,
			int inputFeatureMaps, int activationsStartIndex, int activationsFeatureMapsDistance, int activationsFeatureMapsLength,
			int activationsMiniBatchDistance, int outputStartIndex, int inputFeatureMapsDistance, int n,
			float a, float b);

	public native int BackpropagationConv2DWeightUpdates(int deviceID, int activationID, int weightsID, int weightsUpdatesID, int outputID, int NDrange,
			int miniBatchSize, int weightsRowsDistance, int weightsColumnsDistance, int weightsStartIndex,
			int weightsUpdatesOutputFiltersDistance, int weightsUpdatesInputFiltersDistance, int weightsInputFiltersDistance, int weightsOutputFiltersDistance,
			int weightsUpdatesRowsDistance, int activationStartIndex, int activationFeatureMapsDistance, int activationFeatureMapRowsDistance,
			int activationFeatureMapColumnsDistance, int activationMiniBatchDistance, int outputFeatureMapRows, int outputFeatureMapColumns,
			int outputMiniBatchDistance, int outputStartIndex, int outputFeatureMapsDistance, int outputFeatureMapRowsDistance,
			int outputFeatureMapColumnsDistance, int rowStride, int columnStride,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay);

	public native int FullyConnectedWeightUpdates(int deviceID, int inputID, int weightsID, int weightUpdatesID, int ffActivationID, int NDrange,
			int miniBatchSize, int inputStartPosition, int inputColumnStep, int inputRowStep,
			int activationStartPosition, int activationColumnStep, int activationRowStep, int weightStartPosition,
			int weightsColumns, int weightsRowsDistance, int weightsColumnsDistance,
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay);

	public native int MSEDerivative(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int activationStartPosition, int activationRowStep, int activationColumnStep, int targetStartPosition,
			int targetRowStep, int targetColumnStep, int resultStartPosition, int resultRowStep,
			int resultColumnStep);

	public native int SoftmaxLoss(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int activationStartPosition, int activationRowStep, int activationColumnStep, int targetStartPosition,
			int targetRowStep, int targetColumnStep, int resultStartPosition, int resultRowStep,
			int resultColumnStep, int reverse);

	public native int NegativeLogProbability(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int miniBatchSize, int activationStartPosition, int activationRowStep, int activationColumnStep,
			int targetStartPosition, int targetRowStep, int targetColumnStep, int resultStartPosition,
			int resultRowStep, int resultColumnStep);

	public native int MSE(int deviceID, int activationID, int targetID, int resultID, int NDrange,
			int activationStartPosition, int activationRowStep, int activationColumnStep, int targetStartPosition,
			int targetRowStep, int targetColumnStep, int resultStartPosition, int resultRowStep,
			int resultColumnStep);

	public native int Sigmoid(int deviceID, int ioarrayID, int NDrange, int startIndex);

	public native int Tanh(int deviceID, int ioarrayID, int NDrange, int startIndex);

	public native int ReLU(int deviceID, int ioarrayID, int NDrange, int startIndex);

	public native int SoftReLU(int deviceID, int ioarrayID, int NDrange, int startIndex);

	public native int ReLUDerivative(int deviceID, int resultsID, int activationsID, int NDrange,
			int activationsStartIndex, int resultStartIndex);

	public native int TanhDerivative(int deviceID, int resultsID, int activationsID, int NDrange,
			int activationsStartIndex, int resultStartIndex);

	public native int SigmoidDerivative(int deviceID, int resultsID, int activationsID, int NDrange,
			int activationsStartIndex, int resultStartIndex);

	public native int SoftReLUDerivative(int deviceID, int resultsID, int activationsID, int NDrange,
			int activationsStartIndex, int resultStartIndex);

	public native int Mask(int deviceID, int inputOutputID, int maskArrayID, int NDRange,
			int inputStartIndex, int maskStartIndex);

	public native int prng(int deviceID, int outputID, int NDrange);

	public native int prngRestart(int deviceID);

	public native int prngGaussian(int deviceID, int outputID, int NDrange);

	public native int BernoulliKernel(int deviceID, int outputID, int NDrange);

	public native int Noise(int deviceID, int ioarrayID, int NDrange,
			int activationsStartIndex, float corruptionLevel, float corruptedValue);

	public native int NoiseMask(int deviceID, int ioarrayID, int maskArrayID, int NDrange,
			int activationsStartIndex, int maskStartIndex, float corruptionLevel, float corruptedValue);

	public native int Fill(int deviceID, int arrayID, float value);

	public native int Clear(int deviceID, int arrayID);

	public native int kernelRun(int kernelID);

	public native int kernelRunAsync(int kernelID);

	public native int kernelRunJob(char[] job);

	public native int kernelState(int kernelID);

	public native int kernelPrintProfilingInfo(int kernelID);

	public native int pathSetToCL(char[] path);

	public native int checkLibraryLoad();

	public native int logMessage(char[] message);

	public native int restartLibrary();

	public native int cloneWeightedSum(int N);

	public native int cloneConv2DFF(int N);

	public native int cloneLRN(int N);

	public native int cloneMaxoutFunction(int N);

	public native int cloneBackPropagationMaxout(int N);

	public native int cloneBackpropagationConv2D2(int N);

	public native int cloneMaxPooling2DCC(int N);

	public native int cloneSoftmaxFunction(int N);

	public native int cloneStochasticPooling2DCC(int N);

	public native int cloneAveragePooling2DCC(int N);

	public native int cloneBackpropagationMaxPooling2DCC(int N);

	public native int cloneBackpropagationAveragePooling2DCC(int N);

	public native int cloneBackPropagationLRN(int N);

	public native int cloneBackpropagationConv2DWeightUpdates(int N);

	public native int cloneFullyConnectedWeightUpdates(int N);

	public native int cloneMSEDerivative(int N);

	public native int cloneSoftmaxLoss(int N);

	public native int cloneNegativeLogProbability(int N);

	public native int cloneMSE(int N);

	public native int cloneSigmoid(int N);

	public native int cloneTanh(int N);

	public native int cloneReLU(int N);

	public native int cloneSoftReLU(int N);

	public native int cloneReLUDerivative(int N);

	public native int cloneTanhDerivative(int N);

	public native int cloneSigmoidDerivative(int N);

	public native int cloneSoftReLUDerivative(int N);

	public native int cloneMask(int N);

	public native int cloneNoise(int N);

	public native int cloneNoiseMask(int N);

	public native int cloneFill(int N);

	public native int cloneClear(int N);
}
