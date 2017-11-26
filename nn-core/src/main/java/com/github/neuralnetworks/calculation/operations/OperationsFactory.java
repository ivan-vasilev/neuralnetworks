package com.github.neuralnetworks.calculation.operations;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.TensorFunction.TensorFunctionDerivative;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiAveragePooling2D;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiMask;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiMaxPooling2D;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiMaxout;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiNoise;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiNoiseMask;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiStochasticPooling2D;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiWeightedSumConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.aparapi.ConnectionCalculatorConv;
import com.github.neuralnetworks.calculation.operations.aparapi.LRN;
import com.github.neuralnetworks.calculation.operations.aparapi.ReLU;
import com.github.neuralnetworks.calculation.operations.aparapi.Sigmoid;
import com.github.neuralnetworks.calculation.operations.aparapi.SoftReLU;
import com.github.neuralnetworks.calculation.operations.aparapi.Tanh;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiBackpropagationConv2DWeightUpdates;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiFullyConnectedWeightUpdates;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiMSELossFunction;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiSoftmaxLossFunction;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackPropagationConv2D;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackPropagationFullyConnected;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackPropagationLRN;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackpropagationAveragePooling2D2;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackpropagationMaxPooling2D2;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackpropagationMaxout;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.ReLUDerivative;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.SigmoidDerivative;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.SoftReLUDerivative;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.TanhDerivative;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackpropagationMaxout.AparapiBackpropMaxout;
import com.github.neuralnetworks.calculation.operations.aparapi.random.BernoulliDistribution;
import com.github.neuralnetworks.calculation.operations.cpu.CPUClear;
import com.github.neuralnetworks.calculation.operations.cpu.SoftmaxFunction;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLAveragePooling2DBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLAveragePooling2DConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLBernoulli;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLClear;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLConv2DBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLConv2DBPWeightUpdates;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLConv2DFFConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLFullyConnectedWeightUpdates;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLLRN;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLLRNBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMSE;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMask;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMaxPooling2DBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLMaxPooling2DConnectionCalculator;
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
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLWeightedSumBP;
import com.github.neuralnetworks.calculation.operations.opencl.kernels.OpenCLWeightedSumConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.LossFunction;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Factory for all operations (propagation and testing) over neural networks
 */
public class OperationsFactory
{

	public static ConnectionCalculator weightedSum()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLWeightedSumConnectionCalculator();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiWeightedSumConnectionCalculator();
		}

		return result;
	}

	public static boolean isWeightedSum(ConnectionCalculator c)
	{
		return c != null && (c instanceof AparapiWeightedSumConnectionCalculator || c instanceof OpenCLWeightedSumConnectionCalculator);
	}

	public static ConnectionCalculator sigmoidConnectionCalculator()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLWeightedSumConnectionCalculator();
			((OpenCLWeightedSumConnectionCalculator) result).addActivationFunction(new OpenCLSigmoid());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiWeightedSumConnectionCalculator();
			((AparapiWeightedSumConnectionCalculator) result).addActivationFunction(new Sigmoid());
		}

		return result;
	}

	public static boolean isSigmoidConnectionCalculator(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorImpl && ((ConnectionCalculatorImpl) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorImpl) c).getActivationFunctions().stream().filter(f -> f instanceof Sigmoid || f instanceof OpenCLSigmoid).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator tanhConnectionCalculator()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLWeightedSumConnectionCalculator();
			((OpenCLWeightedSumConnectionCalculator) result).addActivationFunction(new OpenCLTanh());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiWeightedSumConnectionCalculator();
			((AparapiWeightedSumConnectionCalculator) result).addActivationFunction(new Tanh());
		}

		return result;
	}

	public static boolean isTanhConnectionCalculator(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof AparapiWeightedSumConnectionCalculator && ((AparapiWeightedSumConnectionCalculator) c).getActivationFunctions() != null)
			{
				result = ((AparapiWeightedSumConnectionCalculator) c).getActivationFunctions().stream().filter(f -> f instanceof Tanh || f instanceof OpenCLTanh).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator reluConnectionCalculator()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLWeightedSumConnectionCalculator();
			((OpenCLWeightedSumConnectionCalculator) result).addActivationFunction(new OpenCLReLU());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiWeightedSumConnectionCalculator();
			((AparapiWeightedSumConnectionCalculator) result).addActivationFunction(new ReLU());
		}

		return result;
	}

	public static boolean isReLUConnectionCalculator(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorImpl && ((ConnectionCalculatorImpl) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorImpl) c).getActivationFunctions().stream().filter(f -> f instanceof ReLU || f instanceof OpenCLReLU).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator softReLUConnectionCalculator()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLWeightedSumConnectionCalculator();
			((OpenCLWeightedSumConnectionCalculator) result).addActivationFunction(new OpenCLSoftReLU());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiWeightedSumConnectionCalculator();
			((AparapiWeightedSumConnectionCalculator) result).addActivationFunction(new SoftReLU());
		}

		return result;
	}

	public static boolean isSoftReLUConnectionCalculator(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorImpl && ((ConnectionCalculatorImpl) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorImpl) c).getActivationFunctions().stream().filter(f -> f instanceof SoftReLU || f instanceof OpenCLSoftReLU).findAny().isPresent();
			}
		}

		return result;
	}

	public static TensorFunction sigmoidFunction()
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLSigmoid();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new Sigmoid();
		}

		return result;
	}

	public static boolean isSigmoidFunction(TensorFunction f)
	{
		return f != null && (f instanceof Sigmoid || f instanceof OpenCLSigmoid);
	}

	public static TensorFunction tanhFunction()
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLTanh();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new Tanh();
		}

		return result;
	}

	public static boolean isTanhFunction(TensorFunction f)
	{
		return f != null && (f instanceof Tanh || f instanceof OpenCLTanh);
	}

	public static TensorFunction softReLUFunction()
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLSoftReLU();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new SoftReLU();
		}

		return result;
	}

	public static boolean isSoftReLUFunction(TensorFunction f)
	{
		return f != null && (f instanceof SoftReLU || f instanceof OpenCLSoftReLU);
	}

	public static TensorFunction reLUFunction()
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLReLU();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ReLU();
		}

		return result;
	}

	public static boolean isReLUFunction(TensorFunction f)
	{
		return f != null && (f instanceof ReLU || f instanceof OpenCLReLU);
	}

	public static TensorFunction softmaxFunction()
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLSoftmax();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI && result == null)
		{
			result = new SoftmaxFunction();
		}

		return result;
	}

	public static boolean isSoftmaxFunction(TensorFunction t)
	{
		return t != null && (t instanceof SoftmaxFunction || t instanceof OpenCLSoftmax);
	}

	public static TensorFunctionDerivative sigmoidDerivativeFunction()
	{
		TensorFunctionDerivative result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLSigmoidDerivative();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new SigmoidDerivative();
		}

		return result;
	}

	public static boolean isSigmoidDerivativeFunction(TensorFunction f)
	{
		return f != null && (f instanceof SigmoidDerivative || f instanceof OpenCLSigmoidDerivative);
	}

	public static TensorFunctionDerivative tanhDerivativeFunction()
	{
		TensorFunctionDerivative result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLTanhDerivative();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new TanhDerivative();
		}

		return result;
	}

	public static boolean isTanhDerivativeFunction(TensorFunction f)
	{
		return f != null && (f instanceof TanhDerivative || f instanceof OpenCLTanhDerivative);
	}

	public static TensorFunctionDerivative softReLUDerivativeFunction()
	{
		TensorFunctionDerivative result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLSoftReLUDerivative();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new SoftReLUDerivative();
		}

		return result;
	}

	public static boolean isSoftReLUDerivativeFunction(TensorFunction f)
	{
		return f != null && (f instanceof SoftReLUDerivative || f instanceof OpenCLSoftReLUDerivative);
	}

	public static TensorFunctionDerivative reLUDerivativeFunction()
	{
		TensorFunctionDerivative result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLReLUDerivative();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ReLUDerivative();
		}

		return result;
	}

	public static boolean isReLUDerivativeFunction(TensorFunction f)
	{
		return f != null && (f instanceof ReLUDerivative || f instanceof OpenCLReLUDerivative);
	}

	public static ConnectionCalculator softmaxCC()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLWeightedSumConnectionCalculator();
			((AparapiWeightedSumConnectionCalculator) result).addActivationFunction(new OpenCLSoftmax());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiWeightedSumConnectionCalculator();
			((AparapiWeightedSumConnectionCalculator) result).addActivationFunction(new SoftmaxFunction());
		}

		return result;
	}

	public static boolean isSoftmaxCC(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorImpl && ((ConnectionCalculatorImpl) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorImpl) c).getActivationFunctions().stream().filter(f -> f instanceof SoftmaxFunction || f instanceof OpenCLSoftmax).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator maxout()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			// result = OpenCLMaxout(); TODO
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiMaxout();
		}

		return result;
	}

	public static boolean isMaxout(ConnectionCalculator c)
	{
		return c != null && (c instanceof AparapiMaxout);
	}

	public static ConnectionCalculator conv2D()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLConv2DFFConnectionCalculator();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ConnectionCalculatorConv();
		}

		return result;
	}

	public static boolean isConv2D(ConnectionCalculator c)
	{
		return c != null && (c instanceof ConnectionCalculatorConv || c instanceof OpenCLConv2DFFConnectionCalculator);
	}

	public static ConnectionCalculator conv2DReLU()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLConv2DFFConnectionCalculator();
			((OpenCLConv2DFFConnectionCalculator) result).addActivationFunction(new OpenCLReLU());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ConnectionCalculatorConv();
			((ConnectionCalculatorConv) result).addActivationFunction(new ReLU());
		}

		return result;
	}

	public static boolean isConv2DReLU(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorConv && ((ConnectionCalculatorConv) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorConv) c).getActivationFunctions().stream().filter(f -> f instanceof ReLU || f instanceof OpenCLReLU).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator conv2DSigmoid()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLConv2DFFConnectionCalculator();
			((OpenCLConv2DFFConnectionCalculator) result).addActivationFunction(new OpenCLSigmoid());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ConnectionCalculatorConv();
			((ConnectionCalculatorConv) result).addActivationFunction(new Sigmoid());
		}

		return result;
	}

	public static boolean isConv2DSigmoid(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorConv && ((ConnectionCalculatorConv) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorConv) c).getActivationFunctions().stream().filter(f -> f instanceof Sigmoid || f instanceof OpenCLSigmoid).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator conv2DSoftReLU()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLConv2DFFConnectionCalculator();
			((OpenCLConv2DFFConnectionCalculator) result).addActivationFunction(new OpenCLSoftReLU());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ConnectionCalculatorConv();
			((ConnectionCalculatorConv) result).addActivationFunction(new SoftReLU());
		}

		return result;
	}

	public static boolean isConv2DSoftReLU(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorConv && ((ConnectionCalculatorConv) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorConv) c).getActivationFunctions().stream().filter(f -> f instanceof SoftReLU || f instanceof OpenCLSoftReLU).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator conv2DTanh()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLConv2DFFConnectionCalculator();
			((OpenCLConv2DFFConnectionCalculator) result).addActivationFunction(new OpenCLTanh());
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new ConnectionCalculatorConv();
			((ConnectionCalculatorConv) result).addActivationFunction(new Tanh());
		}

		return result;
	}

	public static boolean isConv2DTanh(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			if (c instanceof ConnectionCalculatorConv && ((ConnectionCalculatorConv) c).getActivationFunctions() != null)
			{
				result = ((ConnectionCalculatorConv) c).getActivationFunctions().stream().filter(f -> f instanceof Tanh || f instanceof OpenCLTanh).findAny().isPresent();
			}
		}

		return result;
	}

	public static ConnectionCalculator maxPooling2D()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLMaxPooling2DConnectionCalculator();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiMaxPooling2D();
		}

		return result;
	}

	public static boolean isMaxPooling2D(ConnectionCalculator c)
	{
		return c instanceof AparapiMaxPooling2D || c instanceof OpenCLMaxPooling2DConnectionCalculator;
	}

	public static ConnectionCalculator averagePooling2D()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLAveragePooling2DConnectionCalculator();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiAveragePooling2D();
		}

		return result;
	}

	public static boolean isAveragePooling2D(ConnectionCalculator c)
	{
		return c instanceof AparapiAveragePooling2D || c instanceof OpenCLAveragePooling2DConnectionCalculator;
	}

	public static ConnectionCalculator stochasticPooling2D()
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLStochasticPooling2D();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiStochasticPooling2D();
		}

		return result;
	}

	public static boolean isStochasticPooling2D(ConnectionCalculator c)
	{
		return c != null && (c instanceof AparapiStochasticPooling2D || c instanceof OpenCLStochasticPooling2D);
	}

	public static TensorFunction noise(Tensor inputOutput, float corruptionLevel, float corruptedValue)
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLNoise(corruptionLevel, corruptedValue);
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiNoise(inputOutput, inputOutput.getSize(), corruptionLevel, corruptedValue);
		}

		return result;
	}

	public static boolean isNoise(TensorFunction f)
	{
		return f != null && (f instanceof AparapiNoise || f instanceof OpenCLNoise);
	}

	public static TensorFunction noiseMask(float corruptionLevel, float corruptedValue)
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLNoiseMask(corruptionLevel, corruptedValue);
		}
		
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiNoiseMask(corruptionLevel, corruptedValue);
		}
		
		return result;
	}

	public static boolean isNoiseMask(TensorFunction f)
	{
		return f != null && (f instanceof AparapiNoiseMask || f instanceof OpenCLNoiseMask);
	}

	public static TensorFunction mask(TensorFunction noiseMask)
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLMask((OpenCLNoiseMask) noiseMask);
		}
		
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			AparapiNoiseMask nm = (AparapiNoiseMask) noiseMask;
			result = new AparapiMask(nm);
		}
		
		return result;
	}

	public static boolean isMask(TensorFunction f)
	{
		return f != null && (f instanceof AparapiMask || f instanceof OpenCLMask);
	}

	public static BackPropagationConnectionCalculatorImpl bpConnectionCalculator(Connections c, Properties properties)
	{
		BackPropagationConnectionCalculatorImpl result = null;
		if (c instanceof Conv2DConnection)
		{
			if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
			{
				result = new OpenCLConv2DBP(properties);
			}

			if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
			{
				result = new BackPropagationConv2D(properties);
			}
		} else if (c instanceof FullyConnected)
		{
			if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
			{
				result = new OpenCLWeightedSumBP(properties);
			}

			if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
			{
				result = new BackPropagationFullyConnected(properties);
			}
		}

		return result;
	}

	public static boolean isBPSigmoid(BackPropagationConnectionCalculator c)
	{
		if (c != null && c instanceof ConnectionCalculatorTensorFunctions)
		{
			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getActivationFunctions())
			{
				if (f instanceof SigmoidDerivative || f instanceof OpenCLSigmoidDerivative)
				{
					return true;
				}
			}

			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getInputModifierFunctions())
			{
				if (f instanceof SigmoidDerivative || f instanceof OpenCLSigmoidDerivative)
				{
					return true;
				}
			}
		}

		return false;
	}

	public static boolean isBPReLU(BackPropagationConnectionCalculator c)
	{
		if (c != null && c instanceof ConnectionCalculatorTensorFunctions)
		{
			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getActivationFunctions())
			{
				if (f instanceof ReLUDerivative || f instanceof OpenCLReLUDerivative)
				{
					return true;
				}
			}

			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getInputModifierFunctions())
			{
				if (f instanceof ReLUDerivative || f instanceof OpenCLReLUDerivative)
				{
					return true;
				}
			}
		}

		return false;
	}

	public static boolean isBPSoftReLU(BackPropagationConnectionCalculator c)
	{
		if (c != null && c instanceof ConnectionCalculatorTensorFunctions)
		{
			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getActivationFunctions())
			{
				if (f instanceof SoftReLUDerivative || f instanceof OpenCLSoftReLUDerivative)
				{
					return true;
				}
			}

			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getInputModifierFunctions())
			{
				if (f instanceof SoftReLUDerivative || f instanceof OpenCLSoftReLUDerivative)
				{
					return true;
				}
			}
		}

		return false;
	}

	public static boolean isBPTanh(BackPropagationConnectionCalculator c)
	{
		if (c != null && c instanceof ConnectionCalculatorTensorFunctions)
		{
			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getActivationFunctions())
			{
				if (f instanceof TanhDerivative || f instanceof OpenCLTanhDerivative)
				{
					return true;
				}
			}

			for (TensorFunction f : ((ConnectionCalculatorTensorFunctions) c).getInputModifierFunctions())
			{
				if (f instanceof TanhDerivative || f instanceof OpenCLTanhDerivative)
				{
					return true;
				}
			}
		}

		return false;
	}

	public static BackPropagationConnectionCalculator bpMaxout(Properties properties)
	{
		BackPropagationConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			// TODO
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new BackpropagationMaxout(properties);
		}

		return result;
	}

	public static boolean isBPMaxout(BackPropagationConnectionCalculator c)
	{
		return c != null && (c instanceof AparapiBackpropMaxout || c instanceof BackpropagationMaxout);
	}

	public static BackPropagationConnectionCalculator bpMaxPooling(Properties properties)
	{
		BackPropagationConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLMaxPooling2DBP(properties);
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new BackpropagationMaxPooling2D2(properties);
		}

		return result;
	}

	public static boolean isBPMaxPooling2D(BackPropagationConnectionCalculator c)
	{
		return c instanceof BackpropagationMaxPooling2D2 || c instanceof OpenCLMaxPooling2DBP;
	}

	public static BackPropagationConnectionCalculator bpAveragePooling(Properties properties)
	{
		BackPropagationConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLAveragePooling2DBP(properties);
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new BackpropagationAveragePooling2D2(properties);
		}

		return result;
	}

	public static boolean isBPAveragePooling2D(BackPropagationConnectionCalculator c)
	{
		return c instanceof BackpropagationAveragePooling2D2 || c instanceof OpenCLAveragePooling2DBP;
	}

	public static TensorFunction bernoulliDistribution()
	{
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLBernoulli();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new BernoulliDistribution();
		}

		return result;
	}

	public static ConnectionCalculator lrnConnectionCalculator(float k, int n, float a, float b)
	{
		ConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLLRN(k, n, a, b);
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new LRN(k, n, a, b);
		}

		return result;
	}

	public static boolean isLRNConnectionCalculator(ConnectionCalculator c)
	{
		boolean result = false;
		if (c != null)
		{
			result = c instanceof LRN || c instanceof OpenCLLRN;
		}

		return result;
	}

	public static BackPropagationConnectionCalculator bpLRN(Properties properties, ConnectionCalculator lrn)
	{
		BackPropagationConnectionCalculator result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
		{
			result = new OpenCLLRNBP(properties, (OpenCLLRN) lrn);
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI)
		{
			result = new BackPropagationLRN(properties, (LRN) lrn);
		}

		return result;
	}

	public static boolean hasDropout(ConnectionCalculator cc)
	{
		ConnectionCalculatorImpl fc = (ConnectionCalculatorImpl) cc;
		return fc.getActivationFunctions().stream().filter(f -> isNoiseMask(f)).count() > 0;
	}

	public static LossFunction mse()
	{
		LossFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLMSE();
		}

		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiMSELossFunction();
		}

		return result;
	}

	public static LossFunction softmaxLoss()
	{
		LossFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLSoftmaxLoss();
		}
		
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new AparapiSoftmaxLossFunction();
		}
		
		return result;
	}

	public static TensorFunction clear() {
		TensorFunction result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			result = new OpenCLClear();
		}
		
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			result = new CPUClear();
		}

		return result;
	}

	public static WeightUpdates weightUpdates(WeightsConnections c, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
	{
		WeightUpdates result = null;
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL && result == null)
		{
			if (c instanceof FullyConnected) {
				result = new OpenCLFullyConnectedWeightUpdates((FullyConnected) c, valuesProvider, activations, weightUpdates);
			} else if (c instanceof Conv2DConnection)
			{
				result = new OpenCLConv2DBPWeightUpdates((Conv2DConnection) c, valuesProvider, activations, weightUpdates);
			}
		}
		
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.APARAPI || result == null)
		{
			if (c instanceof FullyConnected) {
				result = new AparapiFullyConnectedWeightUpdates(c, valuesProvider, activations, weightUpdates);
			} else if (c instanceof Conv2DConnection)
			{
				result = new AparapiBackpropagationConv2DWeightUpdates((Conv2DConnection) c, valuesProvider, activations, weightUpdates);
			}
		}

		return result;
	}
}
