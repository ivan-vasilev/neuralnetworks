package com.github.neuralnetworks.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Collection;
import java.util.stream.IntStream;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.WeightsConnections;
import com.github.neuralnetworks.input.FileBatchReader;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.TrainingInputDataImpl;
import com.github.neuralnetworks.training.TrainingInputProvider;

/**
 * Util class
 */
public class Util
{

	public static void fillArray(final float[] array, final float value)
	{
		int len = array.length;
		if (len > 0)
		{
			array[0] = value;
		}

		for (int i = 1; i < len; i += i)
		{
			System.arraycopy(array, 0, array, i, ((len - i) < i) ? (len - i) : i);
		}
	}

	public static void fillArray(final int[] array, final int value)
	{
		int len = array.length;
		if (len > 0)
		{
			array[0] = value;
		}

		for (int i = 1; i < len; i += i)
		{
			System.arraycopy(array, 0, array, i, ((len - i) < i) ? (len - i) : i);
		}
	}

	public static Layer getOppositeLayer(Connections connection, Layer layer)
	{
		return connection.getInputLayer() != layer ? connection.getInputLayer() : connection.getOutputLayer();
	}

	/**
	 * @param layer
	 * @return whether layer is in fact bias layer
	 */
	public static boolean isBias(Layer layer)
	{
		if (layer.getConnections().size() == 1)
		{
			Connections c = layer.getConnections().get(0);
			if (c.getInputLayer() == layer)
			{
				if (c instanceof Conv2DConnection)
				{
					Conv2DConnection cc = (Conv2DConnection) c;
					return cc.getInputFilters() == 1 && cc.getInputFeatureMapRows() == cc.getOutputFeatureMapRows() && cc.getInputFeatureMapColumns() == cc.getOutputFeatureMapColumns();
				} else if (c instanceof FullyConnected)
				{
					FullyConnected cg = (FullyConnected) c;
					return cg.getWeights().getColumns() == 1;
				}
			}
		}

		return false;
	}

	/**
	 * @param layer
	 * @return whether layer is in fact subsampling layer (based on the
	 *         connections)
	 */
	public static boolean isSubsampling(Layer layer)
	{
		Conv2DConnection conv = null;
		Subsampling2DConnection ss = null;
		for (Connections c : layer.getConnections())
		{
			if (c instanceof Conv2DConnection)
			{
				conv = (Conv2DConnection) c;
			} else if (c instanceof Subsampling2DConnection)
			{
				ss = (Subsampling2DConnection) c;
			}
		}

		if (ss != null && (ss.getOutputLayer() == layer || conv == null))
		{
			return true;
		}

		return false;
	}

	/**
	 * @param layer
	 * @return whether layer is in fact convolutional layer (based on the
	 *         connections)
	 */
	public static boolean isConvolutional(Layer layer)
	{
		Conv2DConnection conv = null;
		Subsampling2DConnection ss = null;
		for (Connections c : layer.getConnections())
		{
			if (c instanceof Conv2DConnection)
			{
				conv = (Conv2DConnection) c;
			} else if (c instanceof Subsampling2DConnection)
			{
				ss = (Subsampling2DConnection) c;
			}
		}

		if (conv != null && (conv.getOutputLayer() == layer || ss == null))
		{
			return true;
		}

		return false;
	}

	/**
	 * @param connections
	 * @return whether there is a bias connection in the list
	 */
	public static boolean hasBias(Collection<Connections> connections)
	{
		return connections.stream().filter(c -> isBias(c.getInputLayer())).findAny().isPresent();
	}

	public static void printTensor(float[] array, int rows)
	{
		StringBuilder sb = new StringBuilder();
		NumberFormat formatter = new DecimalFormat("#0.0000");

		IntStream.range(0, rows).forEach(i -> {
			IntStream.range(0, array.length / rows).forEach(j -> sb.append(formatter.format(array[i * array.length / rows + j])).append(array[i * array.length / rows + j] >= 0 ? "  " : " "));
			sb.append(System.getProperty("line.separator"));
		});

		System.out.println(sb.toString());
	}

	public static void inputToFloat(TrainingInputProvider ip, String inputFile, String targetFile)
	{
		try (RandomAccessFile input = inputFile != null ? new RandomAccessFile(inputFile, "rw") : null; RandomAccessFile target = targetFile != null ? new RandomAccessFile(targetFile, "rw") : null)
		{
			byte[] inputBuffer = null;
			if (inputFile != null)
			{
				inputBuffer = new byte[ip.getInputDimensions() * 4];
			}

			byte[] targetBuffer = null;
			if (targetFile != null)
			{
				targetBuffer = new byte[ip.getTargetDimensions() * 4];
			}

			TrainingInputDataImpl ti = new TrainingInputDataImpl(TensorFactory.tensor(1, ip.getInputDimensions()), TensorFactory.tensor(1, ip.getTargetDimensions()));
			for (int i = 0; i < ip.getInputSize(); i++)
			{
				ByteBuffer ibb = inputFile != null ? ByteBuffer.wrap(inputBuffer) : null;
				ByteBuffer tbb = targetFile != null ? ByteBuffer.wrap(targetBuffer) : null;

				ip.populateNext(ti);
				if (inputBuffer != null)
				{
					ibb.asFloatBuffer().put(ti.getInput().getElements());
					input.getChannel().write(ibb);
				}

				if (targetBuffer != null)
				{
					tbb.asFloatBuffer().put(ti.getTarget().getElements());
					target.getChannel().write(tbb);
				}
			}
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	/**
	 * Compute mean value for each element of the dataset
	 * 
	 * @param ip
	 * @param targetFile
	 */
	public static void inputMean(TrainingInputProvider ip, String targetFile)
	{
		double[] doubleMeanValues = new double[ip.getInputDimensions()];

		TrainingInputDataImpl ti = new TrainingInputDataImpl(TensorFactory.tensor(1, ip.getInputDimensions()), TensorFactory.tensor(1, ip.getTargetDimensions()));
		for (int i = 0; i < ip.getInputSize(); i++)
		{
			ip.populateNext(ti);
			for (int j = 0; j < ip.getInputDimensions(); j++)
			{
				doubleMeanValues[j] += ti.getInput().getElements()[j];
			}
		}

		float[] floatMeanValues = new float[ip.getInputDimensions()];
		for (int i = 0; i < ip.getInputDimensions(); i++)
		{
			floatMeanValues[i] = (float) (doubleMeanValues[i] / ip.getInputSize());
		}

		try (RandomAccessFile target = new RandomAccessFile(targetFile, "rw"))
		{
			byte[] dataBuffer = new byte[ip.getInputDimensions() * 4];

			ByteBuffer tbb = ByteBuffer.wrap(dataBuffer);

			tbb.asFloatBuffer().put(floatMeanValues);

			target.getChannel().write(tbb);
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	public static void toFile(String filename, float[] array)
	{
		try (PrintWriter writer = new PrintWriter(filename))
		{
			writer.print(array[0]);
			for (int i = 1; i < array.length; i++)
			{
				writer.print(",");
				writer.print(array[i]);
			}
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}

	public static String networkWeights(NeuralNetwork nn)
	{
		StringBuilder sb = new StringBuilder();

		nn.getConnections()
				.stream()
				.filter(c -> c instanceof WeightsConnections)
				.forEach(
						c -> {
							Tensor w = ((WeightsConnections) c).getWeights();
							if (c instanceof FullyConnected)
							{
								sb.append("FC  ");
							} else
							{
								sb.append("CN  ");
							}

							sb.append("\"" + c.getInputLayer().getName() + "\"->\"" + c.getOutputLayer().getName() + "\"  MIN(" + w.getElements()[TensorFactory.min(w)] + "), MAX("
									+ w.getElements()[TensorFactory.max(w)]
									+ "), AVG(" + TensorFactory.avg(w) + ")"
									+ System.getProperty("line.separator"));
						});

		return sb.toString();
	}

	public static String weightUpdates(NeuralNetwork nn, ValuesProvider weightUpdates)
	{
		StringBuilder sb = new StringBuilder();

		nn.getConnections()
				.stream()
				.filter(c -> c instanceof WeightsConnections)
				.forEach(
						c -> {
							Tensor w = ((WeightsConnections) c).getWeights();
							Tensor wu = weightUpdates.get(c);
							if (c instanceof FullyConnected)
							{
								sb.append("FC  ");
							} else
							{
								sb.append("CN  ");
							}

							int min = TensorFactory.min(wu);
							int max = TensorFactory.max(wu);
							sb.append("\"" + c.getInputLayer().getName() + "\"->\"" + c.getOutputLayer().getName() + "\"  MIN(" + w.getElements()[min] + "->" + wu.getElements()[min] + "), MAX("
									+ w.getElements()[max] + "->" + wu.getElements()[max] + "), AVG(" + TensorFactory.avg(wu) + ")"
									+ System.getProperty("line.separator"));
						});

		return sb.toString();
	}

	public static float[] readFileIntoFloatArray(File file)
	{
		float result[] = null;
		try (RandomAccessFile f = new RandomAccessFile(file, "r"))
		{
			result = new float[(int) (f.length() / 4)];
			FileBatchReader fbr = new FileBatchReader(f, (int) f.length());
			ByteBuffer.wrap(fbr.getNextInput()).asFloatBuffer().get(result);
		} catch (IOException e)
		{
			e.printStackTrace();
		}

		return result;
	}
}
