package com.github.neuralnetworks.builder;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.builder.layer.structure.BiasLayerConnectable;
import com.github.neuralnetworks.builder.layer.structure.LayerBuilder;
import com.github.neuralnetworks.builder.layer.structure.LearnableLayer;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.calculation.operations.OperationsFactory;
import com.github.neuralnetworks.calculation.operations.ClearValuesManager.ClearValuesEventListener;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLManagementListener;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Pair;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * builds a neural network and can also build a trainer
 * 
 * @author tmey
 */
public class NeuralNetworkBuilder
{

	private static final Logger logger = LoggerFactory.getLogger(NeuralNetworkBuilder.class);

	private List<LayerBuilder> listOfLayerBuilder;

	private TrainingInputProvider trainingSet = null;
	private TrainingInputProvider testingSet = null;
	private OutputError error = new MultipleNeuronsOutputError();
	private NNRandomInitializer rand = null; // new NNRandomInitializer(new MersenneTwisterRandomInitializer(0.0001f, 0.01f));

	private List<TrainingEventListener> listOfEventListener = new ArrayList<>();
	private RuntimeConfiguration runtimeConfiguration = null;

	private Boolean overrideAddBiasTo = null;

	// default hyper parameters
	private float learningRate = 0.5f;
	private float momentum = 0;
	private float l1weightDecay = 0;
	private float l2weightDecay = 0;
	private float dropoutRate = 0;

	private int trainingBatchSize = 1;
	private int testBatchSize = 1;
	private int epochs = 1;


	public NeuralNetworkBuilder()
	{
		listOfLayerBuilder = new ArrayList<>();
	}

	public void addLayerBuilder(LayerBuilder layerBuilder)
	{
		if (layerBuilder == null)
		{
			throw new IllegalArgumentException("layerBuilder must be not null!");
		}

		listOfLayerBuilder.add(layerBuilder);
	}

	public void setListOfLayerBuilder(List<LayerBuilder> listOfLayerBuilder)
	{

		if (listOfLayerBuilder == null)
		{
			throw new IllegalArgumentException("listOfLayerBuilder must be not null!");
		}

		for (LayerBuilder builder : listOfLayerBuilder)
		{
			if (builder == null)
			{
				throw new IllegalArgumentException("The list of builders contain a null. This is not allowed!");
			}
		}

		this.listOfLayerBuilder = listOfLayerBuilder;
	}

	/**
	 * create the neural network without a trainer and also ignore the hyper parameters for each layer!
	 * 
	 * @return
	 */
	public NeuralNetworkImpl build()
	{
		return buildNetwork(null);
	}


	private NeuralNetworkImpl buildNetwork(Hyperparameters hyperparameters)
	{
		if (runtimeConfiguration != null)
		{
			Environment.getInstance().setRuntimeConfiguration(runtimeConfiguration);
		}

		NeuralNetworkImpl neuralNetwork = new NeuralNetworkImpl();

		ConnectionFactory cf = new ConnectionFactory();
		neuralNetwork.setProperties(new Properties());
		neuralNetwork.getProperties().setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());
		neuralNetwork.getProperties().setParameter(Constants.CONNECTION_FACTORY, cf);

		neuralNetwork.setLayerCalculator(new LayerCalculatorImpl());

		for (LayerBuilder layerBuilder : listOfLayerBuilder)
		{
			boolean originalBias = true;
			if (overrideAddBiasTo != null && layerBuilder instanceof BiasLayerConnectable)
			{
				originalBias = ((BiasLayerConnectable) layerBuilder).isAddBias();
				((BiasLayerConnectable) layerBuilder).setAddBias(overrideAddBiasTo);
			}

			layerBuilder.build(neuralNetwork, hyperparameters);

			if (overrideAddBiasTo != null && layerBuilder instanceof BiasLayerConnectable)
			{
				((BiasLayerConnectable) layerBuilder).setAddBias(originalBias);
			}
		}

		if (rand != null)
		{
			rand.initialize(neuralNetwork);
		}

		return neuralNetwork;
	}

	/**
	 * create the neural network and a trainer for this network with the given hyper parameters
	 * 
	 * @return
	 */
	public Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>> buildWithTrainer()
	{

		// initialize default hyper parameters
		Hyperparameters hyperparameters = createHyperparameters();

		// create the network
		NeuralNetworkImpl neuralNetwork = buildNetwork(hyperparameters);

		// create the trainer
		Properties properties = createProperties(neuralNetwork);
		properties.setParameter(Constants.HYPERPARAMETERS, hyperparameters);
		properties.setParameter(Constants.BACKPROPAGATION, TrainerFactory.bplc(neuralNetwork, properties));

		Trainer<NeuralNetwork> trainer = new BackPropagationTrainer<>(properties);
		if (Environment.getInstance().getRuntimeConfiguration().getCalculationProvider() == RuntimeConfiguration.CalculationProvider.OPENCL)
		{
			trainer.addEventListener(new OpenCLManagementListener(), 0);
		}

		trainer.addEventListener(new ClearValuesEventListener());

		for (TrainingEventListener trainingEventListener : listOfEventListener)
		{
			trainer.addEventListener(trainingEventListener);
		}

		return new Pair<NeuralNetworkImpl, Trainer<NeuralNetwork>>(neuralNetwork, trainer);
	}

	private Properties createProperties(NeuralNetworkImpl nn)
	{

		Properties p = new Properties();
		p.setParameter(Constants.NEURAL_NETWORK, nn);
		p.setParameter(Constants.TRAINING_INPUT_PROVIDER, trainingSet);
		p.setParameter(Constants.TESTING_INPUT_PROVIDER, testingSet);


		ConnectionFactory cf = new ConnectionFactory();
		p.setParameter(Constants.WEIGHTS_PROVIDER, cf.getWeightsProvider());

		p.setParameter(Constants.WEIGHT_UDPATES, TensorFactory.duplicate(nn.getProperties().getParameter(Constants.WEIGHTS_PROVIDER)));
		p.setParameter(Constants.OUTPUT_ERROR, error);
		p.setParameter(Constants.RANDOM_INITIALIZER, rand);
		p.setParameter(Constants.TRAINING_BATCH_SIZE, trainingBatchSize);
		p.setParameter(Constants.TEST_BATCH_SIZE, testBatchSize);
		p.setParameter(Constants.EPOCHS, epochs);

		LayerCalculatorImpl lc = (LayerCalculatorImpl) nn.getLayerCalculator();
		if (OperationsFactory.isSoftmaxCC(lc.getConnectionCalculator(nn.getOutputLayer())))
		{
			p.setParameter(Constants.LOSS_FUNCTION, OperationsFactory.softmaxLoss());
		} else
		{
			p.setParameter(Constants.LOSS_FUNCTION, OperationsFactory.mse());
		}

		return p;
	}

	private Hyperparameters createHyperparameters()
	{
		Hyperparameters hyperparameters = new Hyperparameters();

		hyperparameters.setDefaultLearningRate(learningRate);
		hyperparameters.setDefaultMomentum(momentum);
		hyperparameters.setDefaultL1WeightDecay(l1weightDecay);
		hyperparameters.setDefaultL2WeightDecay(l2weightDecay);
		hyperparameters.setDefaultDropoutRate(dropoutRate);

		return hyperparameters;
	}

	public boolean resetRandomSeed()
	{
		boolean everythingIsReseeded = true;

		// go throw layer
		int i = 1;
		for (LayerBuilder layerBuilder : listOfLayerBuilder)
		{
			if (layerBuilder instanceof LearnableLayer)
			{
				LearnableLayer learnableLayer = (LearnableLayer) layerBuilder;
				if (learnableLayer.getWeightInitializer() != null)
				{
					if (!learnableLayer.getWeightInitializer().reset())
					{
						everythingIsReseeded = false;
						logger.warn("Can't reset the random seed for " + i + ". layer builder (" + layerBuilder.getClass().getSimpleName() + ")!");
					}
				}
			}

			if (layerBuilder instanceof BiasLayerConnectable)
			{
				BiasLayerConnectable biasLayerConnectable = (BiasLayerConnectable) layerBuilder;
				if (biasLayerConnectable.getBiasWeightInitializer() != null)
				{
					if (!biasLayerConnectable.getBiasWeightInitializer().reset())
					{
						everythingIsReseeded = false;
						logger.warn("Can't reset the random seed for the bias layer of the " + i + ". layer builder (" + layerBuilder.getClass().getSimpleName() + ")!");
					}
				}
			}
			i++;
		}

		// reset default
		if (rand != null)
		{
			if (rand.getBiasRandomInitializer() != null)
			{
				if (!rand.getBiasRandomInitializer().reset())
				{
					everythingIsReseeded = false;
					logger.warn("Can't reset the default random seed for the bias layers!");
				}
			}
			if (rand.getRandomInitializer() != null)
			{
				if (!rand.getRandomInitializer().reset())
				{
					everythingIsReseeded = false;
					logger.warn("Can't reset the default random seed for the main layers!");
				}
			}
		}

		return everythingIsReseeded;
	}

	public void setTrainingSet(TrainingInputProvider trainingSet)
	{
		this.trainingSet = trainingSet;
	}

	public void setTestingSet(TrainingInputProvider testingSet)
	{
		this.testingSet = testingSet;
	}

	public void setError(OutputError error)
	{
		if (error == null)
		{
			throw new IllegalArgumentException("error must be not null!");
		}

		this.error = error;
	}

	public void setRand(NNRandomInitializer rand)
	{
		this.rand = rand;
	}

	public void setLearningRate(float learningRate)
	{

		if (learningRate <= 0)
		{
			throw new IllegalArgumentException("The learning rate must be greater than 0!");
		}

		this.learningRate = learningRate;
	}

	public void setMomentum(float momentum)
	{

		if (momentum < 0)
		{
			throw new IllegalArgumentException("The momentum must be equals or greater than 0!");
		}

		this.momentum = momentum;
	}

	public void setL1weightDecay(float l1weightDecay)
	{
		if (l1weightDecay < 0)
		{
			throw new IllegalArgumentException("The l1weightDecay must be equals or greater than 0!");
		}

		this.l1weightDecay = l1weightDecay;
	}

	public void setL2weightDecay(float l2weightDecay)
	{
		if (l2weightDecay < 0)
		{
			throw new IllegalArgumentException("The l2weightDecay must be equals or greater than 0!");
		}
		this.l2weightDecay = l2weightDecay;
	}

	public void setDropoutRate(float dropoutRate)
	{
		if (dropoutRate < 0)
		{
			throw new IllegalArgumentException("The dropoutRate must be equals or greater than 0!");
		}

		this.dropoutRate = dropoutRate;
	}

	public void setTrainingBatchSize(int trainingBatchSize)
	{
		if (trainingBatchSize < 1)
		{
			throw new IllegalArgumentException("The trainingBatchSize must be minimum 1!");
		}

		this.trainingBatchSize = trainingBatchSize;
	}

	public void setTestBatchSize(int testBatchSize)
	{
		if (testBatchSize < 1)
		{
			throw new IllegalArgumentException("The testBatchSize must be minimum 1!");
		}

		this.testBatchSize = testBatchSize;
	}

	public void setEpochs(int epochs)
	{
		if (epochs < 1)
		{
			throw new IllegalArgumentException("The epochs must be minimum 1!");
		}

		this.epochs = epochs;
	}

	public void setOverrideAddBiasTo(Boolean overrideAddBiasTo)
	{
		this.overrideAddBiasTo = overrideAddBiasTo;
	}

	public float getLearningRate()
	{
		return learningRate;
	}

	public float getMomentum()
	{
		return momentum;
	}

	public float getL1weightDecay()
	{
		return l1weightDecay;
	}

	public float getL2weightDecay()
	{
		return l2weightDecay;
	}

	public float getDropoutRate()
	{
		return dropoutRate;
	}

	public List<LayerBuilder> getListOfLayerBuilder()
	{
		return listOfLayerBuilder;
	}

	public void addEventListener(TrainingEventListener eventListener)
	{
		if (eventListener == null)
		{
			throw new IllegalArgumentException("eventListener must be not null!");
		}

		this.listOfEventListener.add(eventListener);
	}

	public RuntimeConfiguration getRuntimeConfiguration()
	{
		return runtimeConfiguration;
	}

	public void setRuntimeConfiguration(RuntimeConfiguration runtimeConfiguration)
	{
		this.runtimeConfiguration = runtimeConfiguration;
	}

	@Override
	public String toString()
	{

		StringBuffer stringBuffer = new StringBuffer();

		stringBuffer.append("NeuralNetworkBuilder{\nlistOfLayerBuilder={");


		for (LayerBuilder layerBuilder : listOfLayerBuilder)
		{
			stringBuffer.append("\n").append(layerBuilder);
		}

		stringBuffer.append("},\n trainingSet=" + trainingSet +
				", testingSet=" + testingSet +
				", error=" + error +
				", rand=" + rand +
				", overrideAddBiasTo=" + overrideAddBiasTo +
				", learningRate=" + learningRate +
				", momentum=" + momentum +
				", l1weightDecay=" + l1weightDecay +
				", l2weightDecay=" + l2weightDecay +
				", dropoutRate=" + dropoutRate +
				", trainingBatchSize=" + trainingBatchSize +
				", testBatchSize=" + testBatchSize +
				", epochs=" + epochs +
				", eventListener=" + listOfEventListener +
				", runtimeConfiguration=" + runtimeConfiguration +
				'}');


		return stringBuffer.toString();
	}
}
