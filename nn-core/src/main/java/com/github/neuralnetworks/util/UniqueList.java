package com.github.neuralnetworks.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * a simple and slow implementation of unique list
 *
 * @param <E>
 */
public class UniqueList<E> extends ArrayList<E> implements Set<E>
{

	private static final long serialVersionUID = 2378661871806423556L;

	public UniqueList()
	{
		super();
	}

	public UniqueList(Collection<? extends E> arg0)
	{
		super(arg0);
	}

	public UniqueList(int arg0)
	{
		super(arg0);
	}

	@Override
	public boolean add(E e)
	{
		if (!contains(e))
		{
			return super.add(e);
		}

		return false;
	}

	@Override
	public void add(int index, E e)
	{
		if (!contains(e))
		{
			super.add(index, e);
		}
	}

	@Override
	public boolean addAll(Collection<? extends E> c)
	{
		List<? extends E> unique = new ArrayList<>(c);
		for (int i = 0; i < unique.size(); i++)
		{
			if (contains(unique.get(i)))
			{
				unique.remove(i);
			}
		}

		return super.addAll(unique);
	}

	@Override
	public boolean addAll(int index, Collection<? extends E> c)
	{
		List<? extends E> unique = new ArrayList<>(c);
		for (int i = 0; i < unique.size(); i++)
		{
			if (contains(unique.get(i)))
			{
				unique.remove(i);
			}
		}

		return super.addAll(index, unique);
	}
}
