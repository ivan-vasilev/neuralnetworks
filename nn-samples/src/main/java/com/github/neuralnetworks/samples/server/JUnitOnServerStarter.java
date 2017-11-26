package com.github.neuralnetworks.samples.server;

import java.util.List;

import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * This class allow to run a test (that you copy into the function JUnitOnServerStarter#jUnitTest() ) via command line with different epochs on different devices and all or a specific configuration
 * (from JUnitOnServerStarter#runtimeConfigurations()).
 *
 * @author tmey
 */
public abstract class JUnitOnServerStarter
{
	public abstract List<RuntimeConfiguration[]> runtimeConfigurations();

	public abstract void jUnitTest(int epochs, String folder);

	/**
	 *
	 * @param configuration
	 *          0 to test all configurations, between 1 and number of configuration to use a specific configuration
	 * @param epochs
	 *          must be greater than zero!
	 * @param preferredDevice
	 *          <0 to not use it otherwise it is used
	 */
	public void startTests(int configuration, int epochs, int preferredDevice, String folder)
	{
		if (epochs <= 0)
		{
			throw new IllegalArgumentException("number of epochs must be greater than zero!");
		}

		List<RuntimeConfiguration[]> runtimeConfigurations = runtimeConfigurations();

		if (configuration != 0 && configuration < 1 && configuration <= runtimeConfigurations.size())
		{
			throw new IllegalArgumentException("configuration number must be between 1 and " + runtimeConfigurations.size() + " or you don't set it to use all configurations!");
		}


		if (configuration == 0)
		{
			for (int i = 0; i < runtimeConfigurations.size(); i++)
			{

				Environment.getInstance().setRuntimeConfiguration(runtimeConfigurations.get(i)[0]);
				// device!!
				if (preferredDevice >= 0)
				{
					Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().setPreferredDevice(preferredDevice);
				}
				jUnitTest(epochs, folder);
			}
		} else
		{
			Environment.getInstance().setRuntimeConfiguration(runtimeConfigurations.get(configuration - 1)[0]);
			// device!!
			if (preferredDevice >= 0)
			{
				Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().setPreferredDevice(preferredDevice);
			}
			jUnitTest(epochs, folder);
		}


	}

	public void startTestFromCommandLine(String[] args)
	{
		if (args.length < 1 || args.length > 4)
		{
			System.out.println("first argument the epochs, second the configuration (optional), third preferred device (optional, need second parameter)!");
			return;
		}

		int epochs = Integer.parseInt(args[0]);

		int configuration = 0;
		if (args.length >= 2)
		{
			configuration = Integer.parseInt(args[1]);
		}

		int preferredDevice = -1;
		if (args.length >= 3)
		{
			preferredDevice = Integer.parseInt(args[2]);
		}

		String folder = "";
		if (args.length >= 4)
		{
			folder = args[3];
		}

		startTests(configuration, epochs, preferredDevice, folder);
	}


}
