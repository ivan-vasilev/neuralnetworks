package com.github.neuralnetworks.analyzing.network.comparison;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;
import com.github.neuralnetworks.calculation.LayerOrderStrategy;
import com.github.neuralnetworks.calculation.operations.ConnectionCalculatorImpl;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.cpu.ConstantConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.Trainer;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationLayerCalculatorImpl;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Util;

/**
 * @author tmey
 */
public class NetworkActivationAndWeightComparison
{

	private static final Logger logger = LoggerFactory.getLogger(NetworkActivationAndWeightComparison.class);
	private float maxDifference = 0.000001f;
	private File problemFilesDirForVadim = null;
	private boolean compareActivation = true;
	private boolean showDetails = false;

	public void compareTrainedNetworks(BackPropagationTrainer<?> trainer1, BackPropagationTrainer<?> trainer2) throws DifferentNetworksException
	{
		compareTrainedNetworks(trainer1, trainer2, null);
	}

	/**
	 *
	 * @param trainer1
	 * @param trainer2
	 * @param beforeLastBatchWeights
	 *          is needed to save the state of a kernel and rerun the last step. The layer instances must be the one from the network in trainer1 !
	 * @throws DifferentNetworksException
	 */
	public void compareTrainedNetworks(BackPropagationTrainer<?> trainer1, BackPropagationTrainer<?> trainer2, ValuesProvider beforeLastBatchWeights) throws DifferentNetworksException
	{
		ValuesProvider conf1Activations = trainer1.getActivations();
		ValuesProvider conf1Backpropagation = trainer1.getBackpropagation();
		ValuesProvider conf1Weights = ((NeuralNetworkImpl) trainer1.getNeuralNetwork()).getProperties().getParameter(Constants.WEIGHTS_PROVIDER);

		ValuesProvider conf2Activations = trainer2.getActivations();
		ValuesProvider conf2Backpropagation = trainer2.getBackpropagation();
		ValuesProvider conf2Weights = ((NeuralNetworkImpl) trainer2.getNeuralNetwork()).getProperties().getParameter(Constants.WEIGHTS_PROVIDER);

		ComparisonResult comparisonResult = new ComparisonResult();

		if (compareActivation)
		{
			// compare activations

			comparisonResult.setMapActivationDistance(compareActivations(trainer1, conf1Activations, conf2Activations));

			comparisonResult.setMapBackPropagationActivationDistance(compareBackPropagationActivation(trainer1, conf1Backpropagation, conf2Backpropagation));
		}

		comparisonResult.setMapWeightDistance(compareWeights(trainer1, conf1Weights, conf2Weights));

		// analyse result
		{

			List<LayerOrderStrategy.ConnectionCandidate> connectionCandidates = extractConnection((NeuralNetworkImpl) trainer1.getNeuralNetwork());

			if (comparisonResult.getMapActivationDistance() != null && toHighValues(comparisonResult.getMapActivationDistance(), maxDifference))
			{
				Layer firstLayerWithWrongActivation = null;

				for (LayerOrderStrategy.ConnectionCandidate connectionCandidate : connectionCandidates)
				{
					if (comparisonResult.getMapActivationDistance().get(connectionCandidate.target) > maxDifference)
					{
						firstLayerWithWrongActivation = connectionCandidate.target;
						break;
					}
				}

				if (firstLayerWithWrongActivation == null)
				{
					throw new IllegalStateException("Can't find the layer with the high difference between the activations! (Implementation or concept bug)");
				}

				if (this.problemFilesDirForVadim != null)
				{

					if (beforeLastBatchWeights == null)
					{
						logger.warn("Can't provide the state before the run of the kernel because the weights are not provided!");
					} else
					{
						saveForwardPropagationKernelStates(problemFilesDirForVadim, trainer1, firstLayerWithWrongActivation, beforeLastBatchWeights, conf1Backpropagation);
					}
				}

				throw new DifferentNetworksException("the forward activations of this layer are different (max diff. " +
						+comparisonResult.getMapActivationDistance().get(firstLayerWithWrongActivation) + "): \n" + firstLayerWithWrongActivation.toString(), comparisonResult);
			}

			Collections.reverse(connectionCandidates);

			if (comparisonResult.getMapBackPropagationActivationDistance() != null && toHighValues(comparisonResult.getMapBackPropagationActivationDistance(), maxDifference))
			{
				Layer firstLayerWithWrongActivation = null;

				for (LayerOrderStrategy.ConnectionCandidate connectionCandidate : connectionCandidates)
				{
					if (comparisonResult.getMapBackPropagationActivationDistance().get(connectionCandidate.target) != null)
					{
						if (comparisonResult.getMapBackPropagationActivationDistance().get(connectionCandidate.target) > maxDifference)
						{
							firstLayerWithWrongActivation = connectionCandidate.target;
							break;
						}
					} else
					{
						logger.warn("Can't find a max activation difference to this layer and connection: " + connectionCandidate.target.getName() + "\t"
								+ connectionCandidate.connection.getClass().getSimpleName());
					}
				}

				if (firstLayerWithWrongActivation == null)
				{
					throw new IllegalStateException("Can't find the layer with the high difference between the backpropagation activations! (Implementation or concept bug)");
				}

				if (this.problemFilesDirForVadim != null)
				{

					if (beforeLastBatchWeights == null)
					{
						logger.warn("Can't provide the state before the run of the kernel because the weights are not provided!");
					} else
					{
						saveBackwardPropagationKernelStates(problemFilesDirForVadim, trainer1, firstLayerWithWrongActivation, beforeLastBatchWeights, conf1Activations);
					}
				}

				throw new DifferentNetworksException("The backpropagation activations of this layer are different (max diff. "
						+ comparisonResult.getMapBackPropagationActivationDistance().get(firstLayerWithWrongActivation) + "): \n" + firstLayerWithWrongActivation.toString(), comparisonResult);
			}

			// find problem with weight updates
			if (toHighValues(comparisonResult.getMapWeightDistance(), maxDifference))
			{
				Connections firstProblematicConnection = null;

				for (LayerOrderStrategy.ConnectionCandidate connectionCandidate : connectionCandidates)
				{
					if (comparisonResult.getMapWeightDistance().get(connectionCandidate.connection) != null)
					{
						if (comparisonResult.getMapWeightDistance().get(connectionCandidate.connection) > maxDifference)
						{
							firstProblematicConnection = connectionCandidate.connection;
							break;
						}
					} else
					{
						logger.warn("Can't find a max weight difference to this layer and connection: " + connectionCandidate.target.getName() + "\t" + connectionCandidate.connection.getClass().getSimpleName());
					}
				}

				if (firstProblematicConnection == null)
				{
					throw new IllegalStateException("Can't find the connection with the high difference between the weight updates! (Implementation or concept bug)");
				}

				if (this.problemFilesDirForVadim != null)
				{

					if (beforeLastBatchWeights == null)
					{
						logger.warn("Can't provide the state before the run of the kernel because the weights are not provided!");
					} else
					{
						saveUpdateKernelStates(problemFilesDirForVadim, trainer1, firstProblematicConnection, beforeLastBatchWeights, conf1Weights, conf1Activations);
					}
				}

				throw new DifferentNetworksException("The weight update of the connection " + firstProblematicConnection.getClass().getSimpleName()
						+ "  between the layers " + firstProblematicConnection.getInputLayer().getName() + " and " + firstProblematicConnection.getOutputLayer().getName() + " is not correct! (max diff. "
						+ comparisonResult.getMapWeightDistance().get(firstProblematicConnection) + ")", comparisonResult);

			}
		}


	}

	private void saveForwardPropagationKernelStates(File problemFileForVadim, Trainer<?> trainer, Layer layer, ValuesProvider oldWeights, ValuesProvider activation)
	{
		List<Connections> inputConnections = layer.getInputAndOutputConnection().getLeft();
		if (inputConnections.size() != 1)
		{
			logger.warn("Can't save the state of the failing kernel because it has not exact 1 input connection! (instead " + inputConnections.size() + ")");
			return;
		}

		problemFileForVadim.mkdirs();

		Connections connection = inputConnections.get(0);

		// activations
		Util.toFile(problemFileForVadim + File.separator + "input.txt", activation.get(connection.getInputLayer()).getElements());
		Util.toFile(problemFileForVadim + File.separator + "output.txt", getFirstActivation(activation, connection.getOutputLayer()).getElements());

		// weight
		if (connection instanceof WeightsConnections)
		{
			Util.toFile(problemFileForVadim + File.separator + "weights.txt", oldWeights.get(connection).getElements());
		}

		// configuration
		PrintWriter parameters = null;
		try
		{
			parameters = new PrintWriter(problemFileForVadim + File.separator + "parameters.txt");
			ConnectionCalculator connectionCalculator = ((LayerCalculatorImpl) (trainer.getNeuralNetwork()).getLayerCalculator()).getConnectionCalculator(layer);

			List<ConnectionCalculator> inputFunctions = ((ConnectionCalculatorImpl) connectionCalculator).getInputFunctions();
			if (inputFunctions.size() != 1)
			{
				logger.warn("Can't save the kernel properties because there is not exact one kernel for this connection " + connection.getClass().getSimpleName() + "! kernels: " + inputFunctions.size());
			}
			parameters.println(OpenCLCore.getKernelOptionsString((Kernel) inputFunctions.get(0)));
			parameters.close();
		} catch (FileNotFoundException e)
		{
			throw new IllegalStateException("This should not happen!", e);
		}

	}

	private void saveBackwardPropagationKernelStates(File problemFileForVadim, Trainer<?> trainer, Layer layer, ValuesProvider oldWeights, ValuesProvider activation)
	{
		List<Connections> outputConnections = layer.getInputAndOutputConnection().getRight();
		if (outputConnections.size() != 1)
		{
			logger.warn("Can't save the state of the failing kernel because it has not exact 1 output connection! (instead " + outputConnections.size() + ")");
			return;
		}

		problemFileForVadim.mkdirs();

		Connections connection = outputConnections.get(0);

		// activations
		Util.toFile(problemFileForVadim + File.separator + "input.txt", activation.get(connection.getOutputLayer()).getElements());
		Util.toFile(problemFileForVadim + File.separator + "output.txt", getLastActivation(activation, connection.getInputLayer()).getElements());

		// weight
		if (connection instanceof WeightsConnections)
		{
			Util.toFile(problemFileForVadim + File.separator + "weights.txt", oldWeights.get(connection).getElements());
		}

		// configuration
		PrintWriter parameters = null;
		try
		{
			parameters = new PrintWriter(problemFileForVadim + File.separator + "parameters.txt");
			ConnectionCalculator connectionCalculator = ((LayerCalculatorImpl) ((BackPropagationTrainer<?>) trainer).getBPLayerCalculator()).getConnectionCalculator(layer);

			List<ConnectionCalculator> inputFunctions = ((ConnectionCalculatorImpl) connectionCalculator).getInputFunctions();
			if (inputFunctions.size() != 1)
			{
				logger.warn("Can't save the kernel properties because there is not exact one kernel for this connection " + connection.getClass().getSimpleName() + "! kernels: " + inputFunctions.size());
			}
			parameters.println(OpenCLCore.getKernelOptionsString((Kernel) inputFunctions.get(0)));
			parameters.close();
		} catch (FileNotFoundException e)
		{
			throw new IllegalStateException("This should not happen!", e);
		}

	}

	private Tensor getFirstActivation(ValuesProvider activation, Layer layer)
	{
		return activation.getValues().get(layer).get(0);
	}

	private Tensor getLastActivation(ValuesProvider activation, Layer layer)
	{
		List<Tensor> tensorList = activation.getValues().get(layer);

		return tensorList.get(tensorList.size() - 1);
	}

	private void saveUpdateKernelStates(File problemFileForVadim, Trainer<?> trainer, Connections connection, ValuesProvider oldWeights, ValuesProvider newWeights, ValuesProvider activation)
	{

		problemFileForVadim.mkdirs();


		// activations
		Util.toFile(problemFileForVadim + File.separator + "input.txt", activation.get(connection.getOutputLayer()).getElements());
		Util.toFile(problemFileForVadim + File.separator + "output.txt", getLastActivation(activation, connection.getInputLayer()).getElements());

		// weight
		Util.toFile(problemFileForVadim + File.separator + "weights.txt", oldWeights.get(connection).getElements());
		Util.toFile(problemFileForVadim + File.separator + "newWeights.txt", newWeights.get(connection).getElements());

		// configuration
		PrintWriter parameters = null;
		try
		{
			parameters = new PrintWriter(problemFileForVadim + File.separator + "parameters.txt");
			WeightUpdates connectionCalculator = ((BackPropagationTrainer<?>) trainer).getWeightUpdates().get(connection);
			parameters.println(OpenCLCore.getKernelOptionsString((Kernel) connectionCalculator));
			parameters.close();
		} catch (FileNotFoundException e)
		{
			throw new IllegalStateException("This should not happen!", e);
		}

	}

	/**
	 * check if the value of an activation or weight is to high
	 *
	 * @param mapOfDifferences
	 * @param maxDifference
	 * @return
	 */
	private boolean toHighValues(Map<?, Float> mapOfDifferences, float maxDifference)
	{
		for (Float aFloat : mapOfDifferences.values())
		{
			if (aFloat > maxDifference)
			{
				return true;
			}
		}
		return false;
	}

	private List<LayerOrderStrategy.ConnectionCandidate> extractConnection(NeuralNetworkImpl neuralNetwork) {

        Set<Layer> calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(neuralNetwork.getInputLayer());

        neuralNetwork.getLayers().stream().filter(l -> ((LayerCalculatorImpl) neuralNetwork.getLayerCalculator()).getConnectionCalculator(l) instanceof ConstantConnectionCalculator).forEach(l -> {
            calculatedLayers.add(l);
        });

        List<LayerOrderStrategy.ConnectionCandidate> ccc = new BreadthFirstOrderStrategy(neuralNetwork, neuralNetwork.getOutputLayer(), calculatedLayers).order();

        Collections.reverse(ccc);
        return ccc;
    }

	private Map<Connections, Float> compareWeights(BackPropagationTrainer<?> trainer1, ValuesProvider conf1Weights, ValuesProvider conf2Weights) {

        Map<Connections, Float> mapWeightDistance = new HashMap<>();

        logger.info("compare the weights");

        // compare weights
        Iterator<Tensor> itw1 = conf1Weights.getTensors().iterator();
        Iterator<Tensor> itw2 = conf2Weights.getTensors().iterator();

        while (itw1.hasNext() && itw2.hasNext()) {
            float maxValue = 0;

            Tensor t1 = itw1.next();
            Tensor t2 = itw2.next();

            Tensor.TensorIterator t1it = t1.iterator();
            Tensor.TensorIterator t2it = t2.iterator();
            while (t1it.hasNext() && t2it.hasNext()) {
                int t1index = t1it.next();
                int t2index = t2it.next();
                if (t1.getElements() == t2.getElements()) {
                    throw new IllegalStateException("the activation arrays for backpropagation must be different instances!");
                }
                if (t1.getElements()[t1index] == Float.NaN) {
                    throw new IllegalStateException("The first network contains a activation for backpropagation with the value NaN!");
                }
                if (t2.getElements()[t2index] == Float.NaN) {
                    throw new IllegalStateException("The second network contains a activation for backpropagation with the value NaN!");
                }

                float dist = Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]);
                if (maxValue < dist) {
                    maxValue = dist;
                }

                if (dist > maxDifference && showDetails) {
                    Connections c = (Connections) conf1Weights.getKey(t1);

                    WeightUpdates wu = trainer1.getWeightUpdates().get(c);

                    String message = "WU " + c.getInputLayer().getName() + "->" + c.getOutputLayer().getName() + " " + wu.getClass().getSimpleName() + " " + Arrays.toString(t1it.getCurrentPosition()) + "; TARGET->ACTUAL: " + t1.getElements()[t1index]
                            + "->" + t2.getElements()[t2index];

                    logger.error(message);

                }
            }
            Connections c = (Connections) conf1Weights.getKey(t1);

            logger.info("max value difference: " + maxValue + " \t(" + c.getInputLayer().getName() + "->" + c.getOutputLayer().getName() + ")");
            mapWeightDistance.put(c, maxValue);
        }
        return mapWeightDistance;
    }

	private Map<Object, Float> compareBackPropagationActivation(BackPropagationTrainer<?> trainer, ValuesProvider conf1Backpropagation, ValuesProvider conf2Backpropagation) {

        Map<Object, Float> mapActivationError = new HashMap<>();

        logger.info("compare backpropagation activation");

        // compare bp activations
        Iterator<Tensor> itbp1 = conf1Backpropagation.getTensors().iterator();
        Iterator<Tensor> itbp2 = conf2Backpropagation.getTensors().iterator();

        while (itbp1.hasNext() && itbp2.hasNext()) {
            float maxValue = 0;

            Tensor t1 = itbp1.next();
            Tensor t2 = itbp2.next();
            Tensor.TensorIterator t1it = t1.iterator();
            Tensor.TensorIterator t2it = t2.iterator();

            while (t1it.hasNext() && t2it.hasNext()) {
                int t1index = t1it.next();
                int t2index = t2it.next();
                if (t1.getElements() == t2.getElements()) {
                    throw new IllegalStateException("the activation arrays for backpropagation must be different instances!");
                }
                if (t1.getElements()[t1index] == Float.NaN) {
                    throw new IllegalStateException("The first network contains a activation for backpropagation with the value NaN!");
                }
                if (t2.getElements()[t2index] == Float.NaN) {
                    throw new IllegalStateException("The second network contains a activation for backpropagation with the value NaN!");
                }
//                assertTrue((t1.getElements()[t1index] != 0 && t2.getElements()[t2index] != 0) || (t1.getElements()[t1index] == 0 && t2.getElements()[t2index] == 0));

                float dist = Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]);
                if (maxValue < dist) {
                    maxValue = dist;
                }

                if (dist > maxDifference && showDetails) {
                    String message = "BP ";
                    Object key = conf1Backpropagation.getKey(t1);

                    if (key instanceof Layer) {
                        message += ((Layer) key).getName();
                        BackPropagationLayerCalculatorImpl lc = (BackPropagationLayerCalculatorImpl) trainer.getBPLayerCalculator();
                        if (lc.getConnectionCalculator((Layer) key) instanceof BackPropagationConnectionCalculatorImpl) {
                            BackPropagationConnectionCalculatorImpl cc = (BackPropagationConnectionCalculatorImpl) lc.getConnectionCalculator((Layer) key);
                            List<String> functions = new ArrayList<>();
                            for (TensorFunction f : cc.getInputModifierFunctions()) {
                                functions.add(f.getClass().getSimpleName());
                            }

                            for (ConnectionCalculator c : cc.getInputFunctions()) {
                                functions.add(c.getClass().getSimpleName());
                            }

                            for (TensorFunction f : cc.getActivationFunctions()) {
                                functions.add(f.getClass().getSimpleName());
                            }

                            message += " " + functions.stream().collect(Collectors.joining("->"));
                        }
                    } else {
                        message += key.getClass().getSimpleName();
                    }

                    message += Arrays.toString(t1it.getCurrentPosition()) + "; TARGET->ACTUAL: " + t1.getElements()[t1index] + "->" + t2.getElements()[t2index];

                    logger.error(message);
                }
            }

            Object key = conf1Backpropagation.getKey(t1);
            logger.info("max value difference: " + maxValue + " \t(" + ((Layer) key).getName() + ")");
            mapActivationError.put(key, maxValue);
        }
        return mapActivationError;
    }

	private Map<Object, Float> compareActivations(BackPropagationTrainer<?> trainer, ValuesProvider conf1Activations, ValuesProvider conf2Activations) {

        Map<Object, Float> mapActivationError = new HashMap<>();

        logger.info("compare forwardpropagation activation");

        Iterator<Tensor> ita1 = conf1Activations.getTensors().iterator();
        Iterator<Tensor> ita2 = conf2Activations.getTensors().iterator();
        while (ita1.hasNext() && ita2.hasNext()) {
            Tensor t1 = ita1.next();
            Tensor t2 = ita2.next();
            Tensor.TensorIterator t1it = t1.iterator();
            Tensor.TensorIterator t2it = t2.iterator();


            float maxValue = 0;

            while (t1it.hasNext() && t2it.hasNext()) {
                int t1index = t1it.next();
                int t2index = t2it.next();
                if (t1.getElements() == t2.getElements()) {
                    throw new IllegalStateException("the activation arrays must be different instances!");
                }
                if (t1.getElements()[t1index] == Float.NaN) {
                    throw new IllegalStateException("The first network contains a activation with the value NaN!");
                }
                if (t2.getElements()[t2index] == Float.NaN) {
                    throw new IllegalStateException("The second network contains a activation with the value NaN!");
                }
//                assertTrue((t1.getElements()[t1index] != 0 && t2.getElements()[t2index] != 0) || (t1.getElements()[t1index] == 0 && t2.getElements()[t2index] == 0));


                float dist = Math.abs(t1.getElements()[t1index] - t2.getElements()[t2index]);
                if (maxValue < dist) {
                    maxValue = dist;
                }


                if (dist > maxDifference && showDetails) {
                    String message = "ACT ";
                    Object key = conf2Activations.getKey(t2);
                    if (key instanceof Layer) {
                        message += ((Layer) key).getName();
                        LayerCalculatorImpl lc = (LayerCalculatorImpl) trainer.getNeuralNetwork().getLayerCalculator();
                        if (lc.getConnectionCalculator((Layer) key) instanceof ConnectionCalculatorImpl) {
                            ConnectionCalculatorImpl cc = (ConnectionCalculatorImpl) lc.getConnectionCalculator((Layer) key);
                            List<String> functions = new ArrayList<>();
                            for (TensorFunction f : cc.getInputModifierFunctions()) {
                                functions.add(f.getClass().getSimpleName());
                            }

                            for (ConnectionCalculator c : cc.getInputFunctions()) {
                                functions.add(c.getClass().getSimpleName());
                            }

                            for (TensorFunction f : cc.getActivationFunctions()) {
                                functions.add(f.getClass().getSimpleName());
                            }

                            message += " " + functions.stream().collect(Collectors.joining("->"));
                        }
                    } else {
                        message += key.getClass().getSimpleName();
                    }

                    message += Arrays.toString(t1it.getCurrentPosition()) + "; TARGET->ACTUAL: " + t1.getElements()[t1index] + "->" + t2.getElements()[t2index];

                    logger.error(message);
                }
            }
            Object key = conf1Activations.getKey(t1);
            String name = "";
            if (key instanceof Layer) {
                name = ((Layer) key).getName();
            } else {
                name = key.getClass().getSimpleName();
            }
            logger.info("max value difference: " + maxValue + " \t(" + name + ")");
            mapActivationError.put(key, maxValue);
        }
        return mapActivationError;
    }

	public float getMaxDifference()
	{
		return maxDifference;
	}

	public void setMaxDifference(float maxDifference)
	{
		if (maxDifference < 0)
		{
			throw new IllegalArgumentException("The value of the maximum difference between the weights/activation must be larger or equals zero!");
		}
		this.maxDifference = maxDifference;
	}

	public File getProblemFilesDirForVadim()
	{
		return problemFilesDirForVadim;
	}

	public void setProblemFilesDirForVadim(File problemFilesDirForVadim)
	{
		this.problemFilesDirForVadim = problemFilesDirForVadim;
	}

	public boolean isCompareActivation()
	{
		return compareActivation;
	}

	public void setCompareActivation(boolean compareActivation)
	{
		this.compareActivation = compareActivation;
	}

	public boolean isShowDetails()
	{
		return showDetails;
	}

	public void setShowDetails(boolean showDetails)
	{
		this.showDetails = showDetails;
	}
}
