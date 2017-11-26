package com.github.neuralnetworks.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.calculation.NetworkCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.util.RuntimeConfiguration.CalculationProvider;

/**
 * Utility class for serialization/deserialization of objects
 */
public class Serializer
{
	
	/**
	 * Save network to file
	 * 
	 * @param network
	 * @param filename
	 */
	public static void saveNetwork(NeuralNetwork network, String filename)
	{
		saveObject(network, filename);
	}
	
	/**
	 * Save network to stream
	 * 
	 * @param network
	 * @param stream
	 */
	public static void saveNetwork(NeuralNetwork network, OutputStream stream)
	{
		saveObject(network, stream);
	}
	
	/**
	 * load network from file
	 * 
	 * @param filename
	 */
	public static NeuralNetwork loadNetwork(String filename)
	{
		return loadObject(filename);
	}
	
	/**
	 * load network from stream
	 * 
	 * @param stream
	 */
	public static NeuralNetwork loadNetwork(InputStream stream)
	{
		return loadObject(stream);
	}
	
	/**
	 * Save network calculator to file
	 * 
	 * @param calculator
	 * @param filename
	 */
	public static void saveNetworkCalculator(NetworkCalculator<?> calculator, String filename)
	{
		saveObject(calculator, filename);
	}
	
	/**
	 * Save network calculator to stream
	 * 
	 * @param calculator
	 * @param stream
	 */
	public static void saveNetworkCalculator(NetworkCalculator<?> calculator, OutputStream stream)
	{
		saveObject(calculator, stream);
	}
	
	/**
	 * load network calculator from file
	 * 
	 * @param filename
	 */
	public static NetworkCalculator<?> loadNetworkCalculator(String filename)
	{
		return loadObject(filename);
	}
	
	/**
	 * load network calculator from stream
	 * 
	 * @param stream
	 */
	public static NetworkCalculator<?> loadNetworkCalculator(InputStream stream)
	{
		return loadObject(stream);
	}

	/**
	 * Save trainer to file
	 * 
	 * @param trainer
	 * @param filename
	 */
	public static void saveTrainer(Trainer<?> trainer, String filename)
	{
		saveObject(trainer, filename);
	}

	/**
	 * Save trainer to stream
	 * 
	 * @param trainer
	 * @param stream
	 */
	public static void saveTrainer(Trainer<?> trainer, OutputStream stream)
	{
		saveObject(trainer, stream);
	}

	/**
	 * load trainer from file
	 * 
	 * @param filename
	 */
	public static Trainer<?> loadTrainer(String filename)
	{
		return loadObject(filename);
	}

	/**
	 * load trainer from stream
	 * 
	 * @param stream
	 */
	public static Trainer<?> loadTrainer(InputStream stream)
	{
		return loadObject(stream);
	}

	/**
	 * load object from file
	 * 
	 * @param filename
	 */
	public static <T> T loadObject(String filename)
	{
		T result = null;
		try (FileInputStream stream = new FileInputStream(filename))
		{
			result = loadObject(stream);
		} catch (IOException e)
		{
			e.printStackTrace();
		}

		return result;
	}

	/**
	 * load object from stream
	 * 
	 * @param stream
	 */
	@SuppressWarnings("unchecked")
	private static <T> T loadObject(InputStream stream)
	{
		T result = null;
		try (ObjectInputStream ois = new ObjectInputStream(stream))
		{
			result = (T) ois.readObject();
		} catch (IOException | ClassNotFoundException e)
		{
			e.printStackTrace();
		}
		
		return result;
	}

	/**
	 * Save object to file
	 * 
	 * @param object
	 * @param filename
	 */
	public static void saveObject(Object object, String filename)
	{
		try
		{
			Files.deleteIfExists(Paths.get(filename));
		} catch (IOException e1)
		{
			e1.printStackTrace();
		}

		try (FileOutputStream stream = new FileOutputStream(new File(filename)))
		{
			saveObject(object, stream);
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	/**
	 * Save object to stream
	 * 
	 * @param object
	 * @param stream
	 */
	private static <T> void saveObject(T object, OutputStream stream)
	{
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == CalculationProvider.OPENCL)
		{
			OpenCLArrayReferenceManager.getInstance().pushAllToHost();
		}

		try (ObjectOutputStream outputStream = new ObjectOutputStream(stream))
		{
			outputStream.writeObject(object);
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}
