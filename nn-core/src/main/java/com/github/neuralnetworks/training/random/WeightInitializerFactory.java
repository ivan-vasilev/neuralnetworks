package com.github.neuralnetworks.training.random;

import java.sql.Connection;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;

/**
 * @author tmey
 */
public class WeightInitializerFactory
{

	public static void initializeWeights(Connections connection, RandomInitializer randomInitializer)
	{
		if (connection instanceof WeightsConnections)
		{
			WeightsConnections fc = (WeightsConnections) connection;
			randomInitializer.initialize(fc.getWeights());
		}
	}

	public static void initializeWeights(Connection connection, float defaultValue)
    {
        if (connection instanceof WeightsConnections)
        {
            WeightsConnections c = (WeightsConnections) connection;
            c.getWeights().forEach(i -> c.getWeights().getElements()[i] = defaultValue);
        }
    }

	public static boolean isInitialized(Connections connection)
	{
		if (connection instanceof WeightsConnections)
		{
			WeightsConnections c = (WeightsConnections) connection;
			return !isZero(c.getWeights());

		}
		throw new IllegalArgumentException("unknown connection:" + connection.getClass());
	}

	private static boolean isZero(Tensor weights)
	{
		TensorIterator it = weights.iterator();
		while (it.hasNext()) 
		{
			if (weights.getElements()[it.next()] != 0)
			{
				return false;
			}
		}

		return true;
	}
}
