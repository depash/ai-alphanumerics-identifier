"use client"
import Image from "next/image";
import "./main.css"
import React from 'react';
import { useState, useRef } from 'react'

export default function Home() {
  const [brushColor] = useState('black')
  const [brushSize] = useState(10);
  const x = useRef<number>(0)
  const y = useRef<number>(0)
  const drawing = useRef<boolean>(false)

  const onMouseDown = (e: React.MouseEvent) => {
    const canvas = e.target as HTMLCanvasElement;

    const bounding = canvas.getBoundingClientRect();

    const ctx = canvas.getContext("2d")!;

    x.current = e.clientX - bounding.left
    y.current = e.clientY - bounding.top

    const dotSize = brushSize / 2;
    ctx.fillStyle = brushColor;

    ctx.beginPath();
    ctx.arc(x.current, y.current, dotSize, 0, Math.PI * 2, true);
    ctx.fill();

    drawing.current = true
  }

  const onMouseMove = (e: React.MouseEvent) => {
    const canvas = e.target as HTMLCanvasElement;

    const ctx = canvas.getContext("2d")!;

    const bounding = canvas.getBoundingClientRect();

    const newX = e.clientX - bounding.left
    const newY = e.clientY - bounding.top

    if (drawing.current) {
      drawLine(ctx, x.current, y.current, newX, newY);
      x.current = e.clientX - bounding.left
      y.current = e.clientY - bounding.top
    }
  }

  function drawLine(context: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
    context.beginPath();
    context.strokeStyle = brushColor;
    context.lineWidth = brushSize;
    context.lineJoin = "round";
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.closePath();
    context.stroke();
  }

  const onMouseUp = (e: React.MouseEvent) => {
    drawing.current = false
  }

  const onMouseEnter = (e: React.MouseEvent) => {
    const canvas = e.target as HTMLCanvasElement;

    const bounding = canvas.getBoundingClientRect();

    if (drawing.current) {
      x.current = e.clientX - bounding.left
      y.current = e.clientY - bounding.top
    }
  }

  const clearButton = () => {
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d")!;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  return (
    <div className="page-container">
      <canvas
        className="canvas"
        id={'canvas'}
        width={1200}
        height={650}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseEnter={onMouseEnter}
        onMouseDown={onMouseDown}
      />
      <div className="btn-container">
        <button className="guess-btn">
          Guess
        </button>
        <button
          className="guess-btn"
          onClick={clearButton}
        >
          Clear
        </button>
      </div>
    </div>
  );
}
