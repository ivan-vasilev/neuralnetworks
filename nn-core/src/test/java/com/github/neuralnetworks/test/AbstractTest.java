package com.github.neuralnetworks.test;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

import org.junit.AfterClass;

import java.util.Random;

/**
 * Created by chass on 21.11.14.
 */
public abstract class AbstractTest
{
    @AfterClass
    public static void after()
    {
        OpenCLCore.getInstance().finalizeDeviceAll();
    }

	protected enum Runtime
	{
		CPU,
		CPU_SEQ,
		OPENCL;
	}

	protected static void configureGlobalRuntimeEnvironment(Runtime runtime)
	{

		RuntimeConfiguration conf = new RuntimeConfiguration();

		if (runtime.equals(Runtime.CPU))
		{
			conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.APARAPI);
			conf.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
		} else if (runtime.equals(Runtime.CPU_SEQ))
		{
			conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.APARAPI);
			conf.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		} else if (runtime.equals(Runtime.OPENCL))
		{
			conf.setCalculationProvider(RuntimeConfiguration.CalculationProvider.OPENCL);

		}
		conf.setUseDataSharedMemory(false);
		conf.setUseWeightsSharedMemory(false);
		conf.getOpenCLConfiguration().setSynchronizeAfterOpertation(true); // slower but otherwise the results are not pulled back into java object after the operation is executed on opencl
		conf.getOpenCLConfiguration().setAggregateOperations(false); // needs to be false in order to see results after each operation
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	protected boolean isEqual(Tensor t1, Tensor t2)
	{
		if (t1 == null || t2 == null)
			return false;


		if (t1.equals(t2))
			return true;

		if (t1.getDimensions().length != t2.getDimensions().length)
			return false;

		for (int d = 0; d < t1.getDimensions().length; d++)
		{
			if (t1.getDimensions()[d] != t2.getDimensions()[d])
				return false;
		}


		Tensor.TensorIterator it1 = t1.iterator();
		Tensor.TensorIterator it2 = t2.iterator();
		while (it1.hasNext() && it2.hasNext())
		{
			float v1 = t1.getElements()[it1.next()];
			float v2 = t2.getElements()[it2.next()];

			if (Math.abs(v1 - v2) > 0.000001)
				return false;
		}

		return true;

	}

	// set to > 0 to use as constant seed
	protected Random getRandom(long seed)
	{
		// initialize connection weights and input
		Random r = new Random();
		if (seed > 0)
		{
			r.setSeed(seed);
		}

		return r;
	}

	// example config for opencl from ivan
//    oclConf.setCalculationProvider(CalculationProvider.OPENCL);
//    oclConf.setUseDataSharedMemory(false);
//    oclConf.setUseWeightsSharedMemory(false);
//    oclConf.getOpenCLConfiguration().setAggregateOperations(false);
//    oclConf.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
//    oclConf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//    Environment.getInstance().setRuntimeConfiguration(oclConf);

}
