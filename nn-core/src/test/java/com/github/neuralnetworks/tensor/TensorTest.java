package com.github.neuralnetworks.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.stream.IntStream;

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
public class TensorTest extends AbstractTest
{

	public TensorTest(RuntimeConfiguration conf)
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
	public void testTensorOffset()
	{
		// 2x2 unused tensor + a 5x5 parent tensor
		float[] elements = new float[] { 1.1f, 2.1f, 3.1f, 4.1f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
		int offset = 4;
		int[] globalDimensions = new int[] { 5, 5 };
		// upper left, and lower right index of the global dimention
		// parent 3x3 matrix
		Tensor tensor = new Tensor(offset, elements, globalDimensions, new int[][] { { 0, 0 }, { 4, 4 } });

		assertEquals(2, tensor.getDimensions().length);
		assertEquals(5, tensor.getDimensions()[0]);
		assertEquals(5, tensor.getDimensions()[1]);

		assertEquals(offset, tensor.getStartIndex());

		// check parent values
		int count = 1;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				assertEquals(count, tensor.get(i, j), 0);
				count++;
			}
		}

	}

	@Test
	public void testSubTensor()
	{

		// 5x5 parent tensor
		float[] elements = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
		int offset = 0;
		// parent 3x3 matrix
		Tensor parent = new Tensor(offset, elements, new int[] { 5, 5 }, new int[][] { { 0, 0 }, { 4, 4 } });

		assertEquals(2, parent.getDimensions().length);
		assertEquals(5, parent.getDimensions()[0]);
		assertEquals(5, parent.getDimensions()[1]);

		assertEquals(offset, parent.getStartIndex());

		// check parent values
		int count = 1;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				assertEquals(count, parent.get(i, j), 0);
				count++;
			}
		}

		// create suptensor manually which is 3x3 matix in the middel of the 5x5 ,matrix, i.e. offsets 1,1 to 3,3
		Tensor sub1 = new Tensor(parent, new int[][] { { 1, 1 }, { 3, 3 } }, false);

		assertEquals(2, sub1.getDimensions().length);
		assertEquals(3, sub1.getDimensions()[0]);
		assertEquals(3, sub1.getDimensions()[1]);
		assertEquals(6, sub1.getStartIndex());
		assertEquals(18, sub1.getEndIndex());

		// check values
		assertEquals(7, sub1.get(0, 0), 0);
		assertEquals(8, sub1.get(0, 1), 0);
		assertEquals(9, sub1.get(0, 2), 0);
		assertEquals(12, sub1.get(1, 0), 0);
		assertEquals(13, sub1.get(1, 1), 0);
		assertEquals(14, sub1.get(1, 2), 0);
		assertEquals(17, sub1.get(2, 0), 0);
		assertEquals(18, sub1.get(2, 1), 0);
		assertEquals(19, sub1.get(2, 2), 0);

		// create suptensor manually which is 3x3 matix in the upper left corner of the 5x5 ,matrix, i.e. offsets 0,0 to 2,2
		Tensor sub2 = new Tensor(parent, new int[][] { { 0, 0 }, { 2, 2 } }, false);

		assertEquals(2, sub2.getDimensions().length);
		assertEquals(3, sub2.getDimensions()[0]);
		assertEquals(3, sub2.getDimensions()[1]);
		assertEquals(0, sub2.getStartIndex());
		assertEquals(12, sub2.getEndIndex());

		// check values
		assertEquals(1, sub2.get(0, 0), 0);
		assertEquals(2, sub2.get(0, 1), 0);
		assertEquals(3, sub2.get(0, 2), 0);
		assertEquals(6, sub2.get(1, 0), 0);
		assertEquals(7, sub2.get(1, 1), 0);
		assertEquals(8, sub2.get(1, 2), 0);
		assertEquals(11, sub2.get(2, 0), 0);
		assertEquals(12, sub2.get(2, 1), 0);
		assertEquals(13, sub2.get(2, 2), 0);

	}

	@Test
	public void testSubTensorManually01()
	{

		// 5x5 parent tensor
		float[] elements = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
		int offset = 0;
		// create suptensor manually which is 3x3 matix in the middel of the 5x5 ,matrix, i.e. offsets 1,1 to 3,3
		Tensor sub2 = new Tensor(offset, elements, new int[] { 5, 5 }, new int[][] { { 1, 1 }, { 3, 3 } });

		assertEquals(2, sub2.getDimensions().length);
		assertEquals(3, sub2.getDimensions()[0]);
		assertEquals(3, sub2.getDimensions()[1]);
		assertEquals(6, sub2.getStartIndex());
		assertEquals(18, sub2.getEndIndex());

		// check values
		assertEquals(7, sub2.get(0, 0), 0);
		assertEquals(8, sub2.get(0, 1), 0);
		assertEquals(9, sub2.get(0, 2), 0);
		assertEquals(12, sub2.get(1, 0), 0);
		assertEquals(13, sub2.get(1, 1), 0);
		assertEquals(14, sub2.get(1, 2), 0);
		assertEquals(17, sub2.get(2, 0), 0);
		assertEquals(18, sub2.get(2, 1), 0);
		assertEquals(19, sub2.get(2, 2), 0);
	}

	@Test
	public void testSubTensorManually02()
	{

		// 5x5 parent tensor
		float[] elements = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
		int offset = 0;
		// create suptensor manually which is 3x3 matix in the upper left corner of the 5x5 ,matrix, i.e. offsets 0,0 to 2,2
		Tensor sub2 = new Tensor(offset, elements, new int[] { 5, 5 }, new int[][] { { 0, 0 }, { 2, 2 } });

		assertEquals(2, sub2.getDimensions().length);
		assertEquals(3, sub2.getDimensions()[0]);
		assertEquals(3, sub2.getDimensions()[1]);
		assertEquals(0, sub2.getStartIndex());
		assertEquals(12, sub2.getEndIndex());


		// check values
		assertEquals(1, sub2.get(0, 0), 0);
		assertEquals(2, sub2.get(0, 1), 0);
		assertEquals(3, sub2.get(0, 2), 0);
		assertEquals(6, sub2.get(1, 0), 0);
		assertEquals(7, sub2.get(1, 1), 0);
		assertEquals(8, sub2.get(1, 2), 0);
		assertEquals(11, sub2.get(2, 0), 0);
		assertEquals(12, sub2.get(2, 1), 0);
		assertEquals(13, sub2.get(2, 2), 0);
	}

	@Test
	public void testTensorGetElements()
	{
		Tensor tensor = TensorFactory.tensor(3, 3, 3, 3);
		int count = 0;
		for (int a = 0; a < 3; a++)
		{
			for (int b = 0; b < 3; b++)
			{
				for (int c = 0; c < 3; c++)
				{
					for (int d = 0; d < 3; d++)
					{

						tensor.set(count, a, b, c, d);
						count++;
					}
				}
			}
		}

		assertEquals(81, tensor.getSize());

		float[] elements = tensor.getElements();
		assertEquals(81, elements.length);
		for (int i = 0; i < 81; i++)
		{
			assertEquals(i, elements[i], 0);
		}


	}

	@Test
	public void testTensorAccess()
	{
		Tensor tensor = TensorFactory.tensor(3, 3, 3, 3);
		int count = 0;
		for (int a = 0; a < 3; a++)
		{
			for (int b = 0; b < 3; b++)
			{
				for (int c = 0; c < 3; c++)
				{
					for (int d = 0; d < 3; d++)
					{

						tensor.set(count, a, b, c, d);
						count++;
					}
				}
			}
		}

		count = 0;
		for (int a = 0; a < 3; a++)
		{
			for (int b = 0; b < 3; b++)
			{
				for (int c = 0; c < 3; c++)
				{
					for (int d = 0; d < 3; d++)
					{
						assertEquals(count, tensor.get(a, b, c, d), 0);
						count++;
					}
				}
			}
		}
	}

	@Test
	public void testTensorIndex()
	{

		float[] elements = new float[626];
		Tensor tensor = TensorFactory.tensor(elements, 1, 5, 5, 5, 5);
		assertEquals(1, tensor.getStartIndex());
		assertEquals(625, tensor.getEndIndex());

	}

	@Test
	public void testTensorDimentionDistance()
	{

		// second index would be 0 0 0 1 -> the values of the last dimension are 1 afar
		Tensor tensor = TensorFactory.tensor(5, 5, 5, 5);
		assertEquals(125, tensor.getDimensionElementsDistance(0));
		assertEquals(25, tensor.getDimensionElementsDistance(1));
		assertEquals(5, tensor.getDimensionElementsDistance(2));
		assertEquals(1, tensor.getDimensionElementsDistance(3));

	}

	@Test
	public void testTensorIterator()
	{

		Tensor tensor = TensorFactory.tensor(2, 2, 2, 2);

		float[] elements = new float[20];

		for (int i = 0; i < elements.length; i++)
			elements[i] = i;

		tensor.setElements(elements);
		tensor.setStartOffset(4);

		// iterrates over real indexes, i.e. should start with offset
		Tensor.TensorIterator iterator = tensor.iterator();
		int count = 4;
		while (iterator.hasNext())
		{
			Integer index = iterator.next();
			assertNotNull(index);
			assertEquals(count, index.intValue());
			count++;
		}
	}

	@Test
	public void testTensorIterator2()
	{

		Tensor tensor = TensorFactory.tensor(5, 5);

		float[] elements = new float[25];

		for (int i = 0; i < elements.length; i++)
			elements[i] = i;


		tensor.setElements(elements);

		// iterrates over real indexes, i.e. should start with offset
		Tensor.TensorIterator iterator = tensor.iterator(new int[][] { { 2, 2 }, { 3, 3 } });

		assertTrue(iterator.hasNext());
		assertEquals(12, iterator.next().intValue());
		assertTrue(iterator.hasNext());
		assertEquals(13, iterator.next().intValue());
		assertTrue(iterator.hasNext());
		assertEquals(17, iterator.next().intValue());
		assertTrue(iterator.hasNext());
		assertEquals(18, iterator.next().intValue());
		assertTrue(!iterator.hasNext());


	}


	@Test
	public void testGetIndex()
	{

		// 5x5 parent tensor
		float[] elements = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
		int offset = 0;
		// parent 3x3 matrix
		Tensor parent = new Tensor(offset, elements, new int[] { 5, 5 }, new int[][] { { 0, 0 }, { 4, 4 } });

		assertEquals(2, parent.getDimensions().length);
		assertEquals(5, parent.getDimensions()[0]);
		assertEquals(5, parent.getDimensions()[1]);

		assertEquals(offset, parent.getStartIndex());

		// create suptensor manually which is 3x3 matix in the middel of the 5x5 ,matrix, i.e. offsets 1,1 to 3,3
		Tensor subtensor = new Tensor(parent, new int[][] { { 1, 1 }, { 3, 3 } }, false);

		assertEquals(6, subtensor.getIndex(0, 0));
		assertEquals(7, subtensor.getIndex(0, 1));
		assertEquals(8, subtensor.getIndex(0, 2));
		assertEquals(11, subtensor.getIndex(1, 0));
		assertEquals(12, subtensor.getIndex(1, 1));
		assertEquals(13, subtensor.getIndex(1, 2));
		assertEquals(16, subtensor.getIndex(2, 0));
		assertEquals(17, subtensor.getIndex(2, 1));
		assertEquals(18, subtensor.getIndex(2, 2));
	}

	@Test
	public void testTensorComplex()
	{


		Tensor t = TensorFactory.tensor(2, 2, 2);
		float[] elements = t.getElements();

		assertEquals(8, elements.length, 0);

		t.set(1, 0, 0, 0);
		t.set(2, 0, 0, 1);
		t.set(3, 0, 1, 0);
		t.set(4, 0, 1, 1);
		t.set(5, 1, 0, 0);
		t.set(6, 1, 0, 1);
		t.set(7, 1, 1, 0);
		t.set(8, 1, 1, 1);

		Iterator<Integer> it = t.iterator();
		for (int i = 0; i < elements.length && it.hasNext(); i++)
		{
			assertEquals(i + 1, elements[i], 0);
			assertEquals(i + 1, elements[it.next()], 0);
		}

		t = TensorFactory.tensor(5, 5, 5);
		assertEquals(25, t.getDimensionElementsDistance(0), 0);
		assertEquals(5, t.getDimensionElementsDistance(1), 0);
		assertEquals(1, t.getDimensionElementsDistance(2), 0);

		elements = t.getElements();

		for (int i = 0; i < elements.length; i++)
		{
			elements[i] = i + 1;
		}

		Tensor t2 = TensorFactory.tensor(t, new int[][] { { 3, 0, 0 }, { 4, 4, 4 } }, true);
		assertEquals(75, t2.getStartIndex(), 0);
		assertEquals(124, t2.getEndIndex(), 0);
		assertEquals(25, t2.getDimensionElementsDistance(0), 0);
		assertEquals(5, t2.getDimensionElementsDistance(1), 0);
		assertEquals(1, t2.getDimensionElementsDistance(2), 0);
		assertEquals(50, t2.getSize(), 0);
		assertEquals(76, t2.get(0, 0, 0), 0);
		assertEquals(77, t2.get(0, 0, 1), 0);
		assertEquals(81, t2.get(0, 1, 0), 0);
		assertEquals(101, t2.get(1, 0, 0), 0);
		assertEquals(106, t2.get(1, 1, 0), 0);
		assertEquals(112, t2.get(1, 2, 1), 0);

		Tensor[] tarr = TensorFactory.tensor(new int[] { 2, 2, 2 }, new int[] { 3, 3 });
		assertEquals(17, tarr[0].getElements().length, 0);
		assertEquals(0, tarr[0].getStartOffset(), 0);
		assertEquals(8, tarr[1].getStartOffset(), 0);
		assertTrue(tarr[1] instanceof Matrix);

		IntStream.range(0, tarr[0].getElements().length).forEach(i -> tarr[0].getElements()[i] = i + 1);
		assertEquals(7, tarr[0].get(1, 1, 0), 0);
		assertEquals(13, tarr[1].get(1, 1), 0);
	}

	@Test
	// from GeneralTests
	public void testTensor()
	{
		Tensor t = TensorFactory.tensor(2, 2, 2);
		float[] elements = t.getElements();

		assertEquals(8, elements.length, 0);

		t.set(1, 0, 0, 0);
		t.set(2, 0, 0, 1);
		t.set(3, 0, 1, 0);
		t.set(4, 0, 1, 1);
		t.set(5, 1, 0, 0);
		t.set(6, 1, 0, 1);
		t.set(7, 1, 1, 0);
		t.set(8, 1, 1, 1);

		Iterator<Integer> it = t.iterator();
		for (int i = 0; i < elements.length && it.hasNext(); i++)
		{
			assertEquals(i + 1, elements[i], 0);
			assertEquals(i + 1, elements[it.next()], 0);
		}

		t = TensorFactory.tensor(5, 5, 5);
		assertEquals(25, t.getDimensionElementsDistance(0), 0);
		assertEquals(5, t.getDimensionElementsDistance(1), 0);
		assertEquals(1, t.getDimensionElementsDistance(2), 0);

		elements = t.getElements();

		for (int i = 0; i < elements.length; i++)
		{
			elements[i] = i + 1;
		}

		Tensor t2 = TensorFactory.tensor(t, new int[][] { { 3, 0, 0 }, { 4, 4, 4 } }, true);
		assertEquals(75, t2.getStartIndex(), 0);
		assertEquals(124, t2.getEndIndex(), 0);
		assertEquals(25, t2.getDimensionElementsDistance(0), 0);
		assertEquals(5, t2.getDimensionElementsDistance(1), 0);
		assertEquals(1, t2.getDimensionElementsDistance(2), 0);
		assertEquals(50, t2.getSize(), 0);
		assertEquals(76, t2.get(0, 0, 0), 0);
		assertEquals(77, t2.get(0, 0, 1), 0);
		assertEquals(81, t2.get(0, 1, 0), 0);
		assertEquals(101, t2.get(1, 0, 0), 0);
		assertEquals(106, t2.get(1, 1, 0), 0);
		assertEquals(112, t2.get(1, 2, 1), 0);

		Tensor[] tarr = TensorFactory.tensor(new int[] { 2, 2, 2 }, new int[] { 3, 3 });
		assertEquals(17, tarr[0].getElements().length, 0);
		assertEquals(0, tarr[0].getStartOffset(), 0);
		assertEquals(8, tarr[1].getStartOffset(), 0);
		assertTrue(tarr[1] instanceof Matrix);

		IntStream.range(0, tarr[0].getElements().length).forEach(i -> tarr[0].getElements()[i] = i + 1);
		assertEquals(7, tarr[0].get(1, 1, 0), 0);
		assertEquals(13, tarr[1].get(1, 1), 0);
	}


}
