package com.github.neuralnetworks.tensor;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

/**
 * Created by chass on 18.11.14.
 */
@RunWith(Parameterized.class)
public class MatrixTest extends AbstractTest
{

	public MatrixTest(RuntimeConfiguration conf)
	{
		super();
		Environment.getInstance().setRuntimeConfiguration(conf);
	}

	@Parameterized.Parameters
	public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
	{
		RuntimeConfiguration conf1 = new RuntimeConfiguration();
		conf1.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		conf1.setUseDataSharedMemory(false);
		conf1.setUseWeightsSharedMemory(false);

		RuntimeConfiguration conf2 = new RuntimeConfiguration();
		conf2.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
		conf2.setUseDataSharedMemory(true);
		conf2.setUseWeightsSharedMemory(true);

		return Arrays.asList(new RuntimeConfiguration[][] { { conf1 }, { conf2 } });
	}

	@Test
	public void testMatrix01()
	{

		int[] globalDimensions = new int[2];
		// 100 rows, 10 columns
		globalDimensions[0] = 100;
		globalDimensions[1] = 10;


		int[][] globalDimensionsLimit = new int[2][2];
		globalDimensionsLimit[0][0] = 0;
		globalDimensionsLimit[1][0] = 9;
		globalDimensionsLimit[1][1] = 9;
		globalDimensionsLimit[0][1] = 0;

		float[] elements = new float[1000];
		elements[0] = 0.9999999f;
		elements[1] = 0.00000002f;
		for (int i = 1; i < 10; i++)
		{
			elements[i] = 0.00000001f;
		}
	}

	@Test
	public void testMatrix()
	{
		Matrix m = TensorFactory.tensor(5, 6);

		assertEquals(5, m.getRows(), 0);
		assertEquals(6, m.getColumns(), 0);
		assertEquals(6, m.getDimensionElementsDistance(0), 0);
		assertEquals(1, m.getDimensionElementsDistance(1), 0);
		assertEquals(5, m.getDimensions()[0], 0);
		assertEquals(6, m.getDimensions()[1], 0);

		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		assertEquals(2, m.get(0, 1), 0);
		assertEquals(15, m.get(2, 2), 0);

		m = TensorFactory.tensor(1, 6);
		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		assertEquals(2, m.get(0, 1), 0);
		assertEquals(6, m.get(0, 5), 0);

		m = TensorFactory.tensor(6, 1);
		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		assertEquals(2, m.get(1, 0), 0);
		assertEquals(6, m.get(5, 0), 0);

		// submatrix
		Tensor t = TensorFactory.tensor(5, 5, 5);
		float[] elements = t.getElements();

		for (int i = 0; i < elements.length; i++)
		{
			elements[i] = i + 1;
		}

		m = TensorFactory.tensor(t, new int[][] { { 1, 0, 0 }, { 1, 4, 4 } }, true);
		assertEquals(26, m.get(0, 0), 0);
		assertEquals(27, m.get(0, 1), 0);
		assertEquals(36, m.get(2, 0), 0);
		assertEquals(38, m.get(2, 2), 0);

		m = TensorFactory.tensor(t, new int[][] { { 1, 0, 0 }, { 1, 4, 4 } }, true);
		assertEquals(26, m.get(0, 0), 0);
		assertEquals(27, m.get(0, 1), 0);
		assertEquals(36, m.get(2, 0), 0);
		assertEquals(38, m.get(2, 2), 0);

		m = TensorFactory.tensor(t, new int[][] { { 0, 0, 1 }, { 4, 4, 1 } }, true);
		assertEquals(2, m.get(0, 0), 0);
		assertEquals(7, m.get(0, 1), 0);
		assertEquals(12, m.get(0, 2), 0);
		assertEquals(27, m.get(1, 0), 0);
		assertEquals(32, m.get(1, 1), 0);
		assertEquals(37, m.get(1, 2), 0);

		m = TensorFactory.tensor(t, new int[][] { { 2, 2, 1 }, { 3, 3, 1 } }, true);
		assertEquals(62, m.get(0, 0), 0);
		assertEquals(67, m.get(0, 1), 0);
		assertEquals(92, m.get(1, 1), 0);
		Iterator<Integer> it = m.iterator();

		assertEquals(62, m.getElements()[it.next()], 0);
		assertEquals(67, m.getElements()[it.next()], 0);
		it.next();
		assertEquals(92, m.getElements()[it.next()], 0);

		it = m.iterator(new int[][] { { 1, 0 }, { 1, 1 } });
		it.next();
		assertEquals(92, m.getElements()[it.next()], 0);

		m = TensorFactory.tensor(4, 4);
		for (int i = 0; i < m.getElements().length; i++)
		{
			m.getElements()[i] = i + 1;
		}

		Matrix m2 = TensorFactory.tensor(m, new int[][] { { 1, 1 }, { 2, 2 } }, true);
		assertEquals(6, m2.get(0, 0), 0);
		assertEquals(7, m2.get(0, 1), 0);
		assertEquals(10, m2.get(1, 0), 0);
		assertEquals(11, m2.get(1, 1), 0);
	}

}
