package com.github.neuralnetworks.builder.designio;

import java.util.List;

import junit.framework.TestCase;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.analyzing.ConnectionAnalysis;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.CaffeConfigIOUtil;
import com.github.neuralnetworks.builder.layer.ConvolutionalLayerBuilder;
import com.github.neuralnetworks.builder.layer.InputLayerBuilder;
import com.github.neuralnetworks.builder.layer.PoolingLayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.LayerBuilder;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.util.Pair;

public class CaffeConfigIOUtilTest extends TestCase
{
	private static final Logger logger = LoggerFactory.getLogger(CaffeConfigIOUtilTest.class);

	public void testReadBuilder() throws Exception
	{

		// parse the configuration
		NeuralNetworkBuilder neuralNetworkBuilder = CaffeConfigIOUtil.readBuilderFromClasspath("bvlc_caffenet_full_conv.prototxt", "solver.prototxt");

		logger.info("\n\n" + neuralNetworkBuilder.toString() + "\n");

		List<LayerBuilder> listOfLayerBuilder = neuralNetworkBuilder.getListOfLayerBuilder();
		assertEquals(12, listOfLayerBuilder.size());

		Class<?>[] arrayOfLayerBuilderTypes = new Class[] { InputLayerBuilder.class
				, ConvolutionalLayerBuilder.class, PoolingLayerBuilder.class
				, ConvolutionalLayerBuilder.class, PoolingLayerBuilder.class
				, ConvolutionalLayerBuilder.class, ConvolutionalLayerBuilder.class, ConvolutionalLayerBuilder.class, PoolingLayerBuilder.class
				, ConvolutionalLayerBuilder.class, ConvolutionalLayerBuilder.class, ConvolutionalLayerBuilder.class, PoolingLayerBuilder.class };

		for (int i = 0; i < listOfLayerBuilder.size(); i++)
		{
			Class<?> aClass = arrayOfLayerBuilderTypes[i];
			assertTrue("Wrong layer builder at position i: " + listOfLayerBuilder.getClass().getSimpleName()
					+ " instead of " + aClass.getSimpleName()
					, aClass.isInstance(listOfLayerBuilder.get(i)));
		}


		// build the network
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = neuralNetworkBuilder.buildWithTrainer();

		logger.info("\n\n" + ConnectionAnalysis.analyseConnectionWeights(neuralNetworkTrainerPair.getLeft()));

	}

	public void testReadBuilder2() throws Exception
	{

		// parse the configuration
		NeuralNetworkBuilder neuralNetworkBuilder = CaffeConfigIOUtil.readBuilderFromClasspath("train_val_small.prototxt", "solver.prototxt");

		logger.info("\n\n" + neuralNetworkBuilder.toString() + "\n");

		// build the network
		Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> neuralNetworkTrainerPair = neuralNetworkBuilder.buildWithTrainer();

		logger.info("\n\n" + ConnectionAnalysis.analyseConnectionWeights(neuralNetworkTrainerPair.getLeft()));

	}
}