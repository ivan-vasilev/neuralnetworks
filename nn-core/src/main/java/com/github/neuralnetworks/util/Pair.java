package com.github.neuralnetworks.util;

import java.io.Serializable;

/**
 * A Pair class
 */
public class Pair<E, T> implements Serializable
{
	private static final long serialVersionUID = 1035633207383317489L;

	private E left = null;
	private T right = null;

	public Pair(E left, T right)
	{
		super();
	        this.left = left;
		this.right = right;
	}

	public E getLeft()
	{
		return this.left;
	}


	public T getRight()
	{
		return this.right;
	}

}
