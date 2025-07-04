---
title: "How Computers Work: From Sand to Microsoft Word"
date: 2025-06-15T12:00:00-04:00
draft: false
author: "Massil Ait Abdeslam"
tags: [ "beginner"]
description: "An approachable explanation of how computers are built, from basic materials to modern applications."
cover:
  image: "images/computers_explained/cover.jpg"
  alt: "A visual showing layers of computer abstraction"
---

# Introduction

Even though I’ve always been fascinated by computers, before studying them, they felt like magical devices. I had no idea how they actually worked. Like everyone else, I’d heard the classic explanation: “they run on ones and zeroes,” as if that meant anything without context.

But after diving into the subject, I realized something surprising. While computers are incredibly complex machines, their underlying principles are actually quite accessible when approached step by step. Unlike some fields that require years of background knowledge, I genuinely believe anyone with a bit of curiosity and motivation can understand how computers work.

That is what I aim to do with this first post. I want to give you a complete, high-level overview of how a computer works, from sand to the applications you use every day.

# Abstraction: The Heart of Computer Science

You’ve probably heard that computers perform millions, even billions, of operations every second. Maybe you’ve also heard that they contain hundreds of millions of tiny electronic components called transistors. When I first learned that, my reaction was: how does anyone manage all that complexity? Trying to visualize a million of anything feels overwhelming so how can we possibly design a machine that runs a 4K video game at 120 frames per second?

The answer lies in abstraction.

Abstraction is the idea of simplifying complexity by hiding the details you don’t need to see. For example, you can use Microsoft Office without knowing how it was programmed. You trust that the software works and focus only on what it lets you do. That’s abstraction: treating a complex system as a simple tool by relying on the work done underneath it.

In computer science, we apply abstraction at every level. Each layer is built on top of the previous one, using it without needing to fully understand or re-implement it. A software engineer doesn’t have to think about individual transistors, just like you don’t have to think about the programming language used to build Microsoft Office. This is how we manage complexity. It’s also how progress happens, each generation builds on the work of those before. As the saying goes, we are standing on the shoulders of giants.

In the rest of this post, we will explore the key levels of abstraction that make modern computers possible. We’ll start at the very bottom, with sand and electrons and work our way up to the software you use every day. The figure bellow shows the diffrent levels of abstraction that we are going to explore. 

![](/blog/images/computers_explained/abstraction_layer.png)

# Structuring the Journey

At first, the plan seems simple: just climb the abstraction ladder step by step, and we’ll understand how computers work. That approach is valid, but it has a downside. Many things happening at lower levels of abstraction can feel random or disconnected if we don’t first understand how they will be used later on.

To make the learning process as smooth as possible, we’ll use a different strategy.

- Theory (green): This is the foundation of everything. It includes the physics of semiconductors the circuit theory and the mathematics that makes computers possible. Think of it as the essential knowledge required before we can even build a computer.

- Hardware (pink): These are the physical circuits on which everything runs. It includes the CPU, memory, and other components — the parts you can touch and see.

- Software (blue): This is how we tell the hardware what to do. It’s how all the electrical behavior inside the machine is abstracted into human-readable instructions.

- Firmware (purple): This is the middle ground between software and hardware. It controls the hardware at a very low level and helps bridge the two worlds.

So, here’s the game plan: we’ll start with the software, because that’s the way we as humans interact with computers. From there, we’ll work our way down through theory, hardware, and finally firmware to reveal how everything fits together.

# Software

## Application

This is the part you, as a user, actually see and interact with every day. At this level, we think about how an application should look, what features it should have, and how it behaves. This is our end goal all the layers of abstraction exist to make this possible.

Applications come in many forms. They include your web browser, command-line tools (like the ones hackers use in movies), and cutting-edge multiplayer video games with realistic graphics. One thing I want to take a moment to explain is how a computer can even generate a graphical interface. How can it show us visual information on a screen?

The answer lies in how our eyes work. Our vision is based on sensitivity to three primary colors: red, green, and blue. Every color we perceive is just a mix of these three. Computers take advantage of this by representing each color as a combination of red, green, and blue values, this is called RGB color.

To display an image, the computer divides the screen into a grid of very small squares, called pixels. Each pixel is one solid color. By carefully choosing the RGB value for each pixel, the computer can display any image. For example, a resolution like 720p or 1080p refers to the number of pixels in the image, 720p typically means 1280(width) × 720(height) pixels.

If you zoom in enough on a screen, you can actually see the individual RGB components of each pixel. The image below shows what pixels look like close up, three vertical strips of red, green, and blue per pixel.

<img src="/blog/images/computers_explained/pixels.jpg" alt="Alt text" width="50%" style="display: block; margin: auto; padding-bottom: 20px;">

Now, how does the computer communicate this to the screen? It’s pretty simple. Each of the red, green, and blue values is represented by a number between 0 and 255:

- 0 means the color is completely off,
- 255 means it's fully on.

So for example, the color (128, 23, 255) means:

- Red is at about 50% brightness (128 / 255),
- Green is very dim,
- Blue is fully on.

This combination gives us a kind of purple. You can experiment with color combinations using this [color picker](https://www.w3schools.com/colors/colors_picker.asp). Try entering rgb(128, 23, 255) to see the result.

By sending each pixel’s position (like (x, y) coordinates, where (0,0) is the top-left of the screen) and its RGB color values, the computer can create any graphical interface imaginable and render it on your screen.

## Programming Languages
Now that we understand how a computer can display things on the screen, the next question is: how do we tell it what to display and how the application should behave? That’s where programming languages come in.

Let’s take a simple example from Microsoft Word, the word count feature. As a human, how do you count the number of words in a text? Most of the time, you scan the sentence and count the spaces and apostrophes. For simplicity, let’s assume the punctuation is used correctly and there are no double spaces. You could describe your approach as a clear set of steps: go through each character in the text, and each time you see a space or an apostrophe, increase the word count by one.

What you just did is break the task into a list of unambiguous steps. That’s called an algorithm a clear and finite sequence of instructions that solves a problem when followed precisely.

### Computers Are Algorithm Machines

Now we’re getting to the heart of what a computer really is. Although we often say that a computer runs applications, at its core, a computer is a machine designed to follow algorithms. We use those algorithms to build the applications we use every day.

But there’s something important to keep in mind: computers are not “smart.” They don’t understand things the way we do. All they can do is follow instructions, precisely, quickly, and without ever getting tired. Even with recent advances in artificial intelligence, what may look like thinking is still the result of following very complex (but ultimately deterministic) instructions. AI is a fascinating topic, but that’s for another post.

To make a computer follow an algorithm, we need a way to express it in a form the machine can understand. This is what programming languages are for.

### What Is a Programming Language?	

A programming language is a way to write instructions for a computer without ambiguity. Unlike natural languages like English or French, which can be interpreted differently depending on context, programming languages are strict and precise. Every statement has a single, well-defined meaning. The rules of the language called syntax are designed so that the computer always knows exactly what to do.

There are many programming languages, each with their strengths and weaknesses. Some are easy to write but run slower. Others are more difficult to write but run much faster. Despite their differences, they all let us express algorithms in a clear and structured way.

Let’s return to our word count algorithm. In plain English, we might write:

1.	Iterate over every character in the text.
2.	If the character is a space or an apostrophe, increment the word count.

In a programming language, this could look like:

```python
word_count = 0
for character in text:
    if character == " " or character == "'":
        word_count = word_count + 1
return word_count
```

There’s no room for interpretation. The computer will execute these instructions exactly as written.

Every feature in an application, whether it’s changing the font in Word, sorting files, or playing a video, is made up of lots of little algorithms like this one. For example:

- If the user clicks on the "Font" button → show the list of fonts.
- If the user selects "Arial" → update the font of the selected text.

So in summary: computers execute algorithms, and we use programming languages to describe those algorithms in a way the computer can follow.

## Compiler

![](/blog/images/computers_explained/binary.png)

This is the point in the abstraction stack where we make one of the biggest leaps. From here on, we get closer to how the machine actually works. Most software engineers never have to go deeper than this.

Until now, we haven’t mentioned the famous “ones and zeros” that people often associate with computers. We’ve said that programming languages allow us to communicate instructions (algorithms) to the machine, but those languages are still human-readable. A developer can look at a snippet of code and understand what it’s doing as long as they know the language.

But the truth is, computers don’t understand programming languages directly. What they do understand is binary: a sequence of ones and zeros. In reality, these 1s and 0s are implemented as electrical signals, typically, current flowing represents a 1, and no current represents a 0. We’ll dive into the physical meaning of that later when we get into hardware and theory.

So how do we get from programming language to binary code ? That’s the job of the compiler. A compiler is a special program that takes the code we write and translates it into machine code, a long stream of binary instructions that the computer can actually run. That’s exactly why programming language syntax is so strict. It’s because the code needs to go through a compiler and be transformed into machine code. The structure and clarity of the code are what allow the compiler to make that translation properly. These binary instructions are what the CPU executes directly, feeding them into the physical circuits that perform computations.

I know that this process may feel a bit abstract right now but we’ll revisit what machine code looks like and how it interacts with hardware later in this post. For now, just remember this: The compiler takes human-readable code and turns it into machine-readable binary instructions. In the next sections, we’ll explore how those ones and zeros can actually make physical circuits behave intelligently.

## Operating system

This is a layer that most introductions to computers tend to skip, but I think it's important to include. Understanding what an operating system is and what it does gives a more complete picture of how computers actually work. Also, it's something we all interact with, whether we realize it or not. Every computer, smartphone, smartwatch, and even some cars have one. The most well-known operating system is probably Windows by Microsoft. But what exactly is its purpose?

Previously, we saw that applications are built using programming languages, and that they must be compiled into machine code to be run by the computer. But what happens when we want to run multiple applications at the same time? What makes it possible for your files to still be there after you restart your computer? How does the computer manage all of this? That’s exactly the role of the operating system.

An operating system (OS) is itself a program (an application) but one with a special job: it manages the computer’s resources and coordinates other applications. You can think of it as the master application. It decides how much computing power each program gets, controls access to memory, manages files and folders, and handles communication with your screen, keyboard, mouse, and more. The OS is what lets different programs share the computer without stepping on each other’s toes. It's also the layer that stores and organizes files on your hard drive or SSD, making sure everything is saved properly.

So in short: The operating system is a special program that manages the computer’s hardware and helps other programs run smoothly.
In the next sections, we’ll switch gears. We’ll go back to the lowest layers, the physical and logical foundations and climb back up from there to see how it all connects.

# Theory and Hardware

![](/blog/images/computers_explained/hardware.jpg)

## Introduction

Until now, we’ve mostly looked at how we talk to the computer through programming languages, compilers, and operating systems. But we haven’t yet answered the deeper question: how does the computer actually do what we tell it to do? We said that computers are algorithmic machines. They follow a list of instructions written in a programming language, compiled into machine code (a binary file) that the computer can understand. But the real mystery is this: how can a computer do so much using only ones and zeros? That’s where the magic really starts.

In the next sections, we’ll dig into how a computer can take a series of ones and zeros and perform all kinds of operations on them. To understand this, we’ll first explore the mathematical foundation behind computing: Boolean algebra. From there, we’ll see how we can build electrical circuits that implement Boolean logic using a basic electronic component called the transistor. Then we’ll show how simple logic gates can be combined into more complex circuits that can do meaningful work like arithmetic, memory storage, and decision-making.

Finally, we’ll connect the hardware back to the software, and show how they interact with each other. This will complete our tour of the abstraction stack, and hopefully give you a solid and intuitive understanding of how a computer really works — from sand to software.

## Boolean algebra

Boolean algebra might feel a bit abstract and strange at first, but don’t forget our end goal: we want to build circuits that can perform complex operations using only ones and zeros. Boolean algebra gives us exactly the mathematical foundation to do that. Just a side note here, I’m presenting our goal as building circuits using ones and zeros, but in reality, the reason we use ones and zeros comes from Boolean algebra itself. I just think it’s easier to understand if we take the problem in reverse.
Boolean algebra introduces three basic logical operations:

- NOT
- AND
- OR

These operations let us express logic formally. They take binary input (1 or 0) and give a binary output. The behavior of these logic operations is often shown using something called a truth table, a table that lists all possible input combinations and their corresponding output. The truth table for the 3 basic operations are shown bellow:

<div style="overflow-x: auto;">
  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1.5rem; font-size: 14px;">

  <div style="flex: 0 0 auto;">
    <table border="1" style="min-width: 100px; border-collapse: collapse;">
      <caption><strong>NOT</strong></caption>
      <thead>
        <tr>
          <th style="text-align: center; vertical-align: middle;">A</th>
          <th style="text-align: center; vertical-align: middle;">¬A</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="text-align: center;">0</td><td style="text-align: center;">1</td></tr>
        <tr><td style="text-align: center;">1</td><td style="text-align: center;">0</td></tr>
      </tbody>
    </table>
  </div>

  <div style="flex: 0 0 auto;">
    <table border="1" style="min-width: 140px; border-collapse: collapse;">
      <caption><strong>AND</strong></caption>
      <thead>
        <tr>
          <th style="text-align: center; vertical-align: middle;">A</th>
          <th style="text-align: center; vertical-align: middle;">B</th>
          <th style="text-align: center; vertical-align: middle;">A ∧ B</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="text-align: center;">0</td><td style="text-align: center;">0</td><td style="text-align: center;">0</td></tr>
        <tr><td style="text-align: center;">0</td><td style="text-align: center;">1</td><td style="text-align: center;">0</td></tr>
        <tr><td style="text-align: center;">1</td><td style="text-align: center;">0</td><td style="text-align: center;">0</td></tr>
        <tr><td style="text-align: center;">1</td><td style="text-align: center;">1</td><td style="text-align: center;">1</td></tr>
      </tbody>
    </table>
  </div>

  <div style="flex: 0 0 auto;">
    <table border="1" style="min-width: 140px; border-collapse: collapse;">
      <caption><strong>OR</strong></caption>
      <thead>
        <tr>
          <th style="text-align: center; vertical-align: middle;">A</th>
          <th style="text-align: center; vertical-align: middle;">B</th>
          <th style="text-align: center; vertical-align: middle;">A ∨ B</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style="text-align: center;">0</td><td style="text-align: center;">0</td><td style="text-align: center;">0</td></tr>
        <tr><td style="text-align: center;">0</td><td style="text-align: center;">1</td><td style="text-align: center;">1</td></tr>
        <tr><td style="text-align: center;">1</td><td style="text-align: center;">0</td><td style="text-align: center;">1</td></tr>
        <tr><td style="text-align: center;">1</td><td style="text-align: center;">1</td><td style="text-align: center;">1</td></tr>
      </tbody>
    </table>
  </div>

</div>
</div>
So this is what each operation does:

- NOT flips the input (0 becomes 1, and 1 becomes 0).
- AND outputs 1 only if both inputs are 1.
- OR outputs 1 if at least one input is 1.

Boolean algebra can be used to formalize logic statements. For example, let’s take a real-world sentence:"The light turns on if the switch is on or the switch is on and the door is closed." We can translate this into Boolean algebra. Let:

- S = the switch is on
- D = the door is closed
- L = the light is on

The expression becomes:
$$L = S  ∨ (S ∧ D)$$

Using Boolean algebra rules (which we won’t go into here), this can be simplified to:

$$L=S$$

So the light is on if the switch is on which makes sense when you think about it. The point here is that we used pure logic to simplify a statement using only symbols. And importantly, the variables S, L, and D could represent anything the logic still applies. This is what Boolean algebra was originally made for: to formalize and simplify logical reasoning.

Now, you might ask: what does any of this have to do with computers? That’s a fair question. At first, Boolean algebra really was a niche topic. It was developed in the 1850s by George Boole, mainly for philosophical logic. It wasn’t until almost 100 years later, in 1937 that Claude Shannon, in his master’s thesis, showed that Boolean algebra could be used to model and analyze electrical switch circuits. At the time, engineers were already building circuits that could perform logical tasks for example: "If signal A and signal B are on, then turn on the motor." Some circuits were even being used to perform arithmetic operations like addition or subtraction. What Shannon showed was that these circuits could be fully analyzed and optimized using Boolean algebra. This realization was groundbreaking. It laid the foundation for digital circuits  circuits that work with binary signals, like current or no current and ultimately led to the development of modern computers.

## Semiconductor Physics and Transistors

One of the classic things people say when talking about computers is that they're made out of sand. That may sound strange at first, but it's actually mostly true. That's because sand is made of silicon dioxide (SiO₂) and if you've heard of Silicon Valley, that's exactly where the name comes from. It refers to silicon, the atomic element found in sand. Silicon has a very special property: it's a semiconductor. You’re probably already familiar with conductors like copper, which allow electricity to flow freely, and insulators like glass, which do not conduct electricity at all. Semiconductors are somewhere in between, under certain conditions they can conduct electricity, and under other conditions they don’t. Silicon is the most commonly used material with this property.

Without getting into the deep physics behind how semiconductors work, we’ll focus on one of the most important inventions in human history: the transistor. A transistor is a tiny electronic component made from semiconductor material (usually silicon) that acts like a switch. But unlike a light switch, which you flip by hand, a transistor is switched electrically  using another small current. It has three pins:

- one for the input,
- one for the output,
- and one for control (sometimes called the gate or base, depending on the type of transistor).

If a current is applied to the control pin, the transistor allows current to flow from input to output. If there's no current at the control pin, the input is disconnected from the output. Transistors are possibly the most produced object in human history. Modern high-end computer chips contain billions of them, and we manufacture trillions every year. 

Now, you might already sense a connection: transistors act like binary switches, and Boolean algebra is all about binary logic. That’s exactly what we’re going to explore in the next section — how we can use transistors to physically implement logic gates and turn Boolean expressions into working circuits.

## Logic gates
![](/blog/images/computers_explained/gates.png)

Now that we’ve covered the background theory, we’re ready to tackle the first real brick of computing: logic gates. Remember our basic Boolean operations: NOT, AND, and OR? Well, we can implement these operations physically using transistors! That means Boolean algebra is not just abstract math anymore, it becomes something we can build with.

Once we can physically represent NOT, AND, and OR, we can construct all kinds of useful circuits. For example, take a simple rule like: “If signal A and signal B are on, then do C.” We now have a way to build a circuit that behaves exactly like that.

Just a side note to highlight why transistors were such a revolutionary invention: The very first computers didn’t use transistors, they used vacuum tubes. These also acted like binary switches, but they were big, expensive, and unreliable. When transistors came along, they allowed for miniaturization (so computers could be smaller), more switches (so they could be more powerful), and better reliability. Transistors enabled the explosion of modern computing and helped shape the digital world we live in today.

With just these three basic gates NOT, AND, and OR, we can start building circuits that perform real computations. In the next sections, we’ll explore how we can combine them to create more advanced logic, arithmetic, and memory circuits.

## Logic circuits

For this section, I had a real dilemma: how technical should I get without going too technical? I believe the concepts here can be understood without diving into every implementation detail, but at the same time, seeing how the circuits are actually built gives the explanation more depth. It makes everything feel less magical when you can see how it really works.

On the other hand, this article is already getting quite long, and I don’t want to overload it. So here’s the compromise I landed on: In this article, I’ll stick to the concepts, and I’ll write a separate post later that shows how we can build these circuits step by step. I’ll link it here once it's finished. For now, let’s just look at some examples of what we can build using logic gates, without going into the exact wiring.

### Number representation

Before we move further into logic circuits, it’s important to address something fundamental: How do computers represent numbers in the first place? Computers operate entirely on numbers. Everything : images, text, video, and even code is stored internally as numbers. For example, we saw earlier that images are just grids of RGB values, which are numeric. But what about text, like the one you’re reading right now? Text is also stored as numbers. The most basic system used for this is called ASCII, which assigns a number between 0 and 255 to each letter, digit, and punctuation mark. For example, the capital letter A is represented by the number 65.

So the next question is: how can a binary machine represent these numbers? We’re used to base 10, where we use digits 0 to 9 to represent values. For example, the number 437 is interpreted as:

$$ 4×10^2+ 3×10^1+ 7×10^0 $$

In binary (base 2), we use only the digits 0 and 1. The exact same principle applies, but we use powers of 2 instead of 10. So a binary number like 1011 means:

$$1×2^3+ 0×2^2+ 1×2^1+ 1×2^0$$
$$= 8 + 0 + 2 + 1 = 11$$

The table below shows the binary representation of the first 8 decimal numbers:

<div style="display: flex; justify-content: center; margin: 1em 0;">

  <table style="border-collapse: collapse; text-align: center;">
    <thead>
      <tr>
        <th style="padding: 8px;">Decimal</th>
        <th style="padding: 8px;">Expanded Expression</th>
        <th style="padding: 8px;">Binary</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>0</td><td>0×2² + 0×2¹ + 0×2⁰</td><td>000</td></tr>
      <tr><td>1</td><td>0×2² + 0×2¹ + 1×2⁰</td><td>001</td></tr>
      <tr><td>2</td><td>0×2² + 1×2¹ + 0×2⁰</td><td>010</td></tr>
      <tr><td>3</td><td>0×2² + 1×2¹ + 1×2⁰</td><td>011</td></tr>
      <tr><td>4</td><td>1×2² + 0×2¹ + 0×2⁰</td><td>100</td></tr>
      <tr><td>5</td><td>1×2² + 0×2¹ + 1×2⁰</td><td>101</td></tr>
      <tr><td>6</td><td>1×2² + 1×2¹ + 0×2⁰</td><td>110</td></tr>
      <tr><td>7</td><td>1×2² + 1×2¹ + 1×2⁰</td><td>111</td></tr>
    </tbody>
  </table>

</div>

This method allows us to represent any number in binary and since everything in a computer can be encoded as numbers, we now have a way to represent anything: text, images, sounds, and more.

### Arithmetic circuit and logic circuits.

Now that we know how to represent numbers in binary and by extension, almost anything, we may wonder how to manipulate these numbers and perform operations on them. That’s exactly where arithmetic circuits come into play.

Using basic logic gates, we can build circuits that perform mathematical operations. For example, An adder takes two binary numbers and adds them. A subtractor, multiplier, and even divider can also be built from logic gates. These circuits let us perform calculations much faster than any human could do manually.

Just a small clarification here: up until now, we’ve only talked about positive integers. In reality, computers can also represent negative numbers and real numbers (like 3.14), but these require different encoding formats, outside the scope of this article.

Now let’s look at a few important logic circuits beyond arithmetic : 

- Multiplexer (MUX): This circuit works like a simple if-statement: If C = 0, output A. If C = 1, output B. Multiplexers are essential for making decisions and enabling conditional behavior in circuits. 

- Comparator: This circuit compares two binary inputs and tells us whether they are equal, greater than, or less than each other. For example, if we compare the binary representation of two text strings, a comparator can tell us if they're the same or different.

With arithmetic and logic circuits, we now have the tools to perform calculations, make decisions, and compare data, the foundation of any computer program.

### Memory circuit

There’s one more crucial component we need to cover: memory. Computers constantly need to store information. Sometimes this means long-term storage, like saving a file. But it also includes short-term memory, like remembering intermediate results during calculations. For example, if I ask you to compute (365 + 25) / 10, you’ll first calculate 365 + 25 = 390, remember 390, then divide by 10 to get 39. That temporary storage of 390 is exactly what a computer needs too and that’s what memory circuits are for. They also of course need long term memory to save file for later use. 

Since everything on a computer is stored as numbers, including this very text file, we need circuits that can store binary values. And yes, we can build such circuits using logic gates. Without diving too deep into the technical details, the basic idea is that logic gates can be arranged in specific configurations to create storage cells that remember binary values.

One important note: not all memory in computers is built from just logic gates. Many memory systems are actually built from other electrical components especially those designed for large capacity. Logic gate-based memory is used where speed is critical but in counterpart they can’t store large binary value. 

With arithmetic, logic, and memory circuits in place, we now have all the core components we need to build a basic computer. That’s exactly what we’ll explore in the next section.

## Cpu and memory

![](/blog/images/computers_explained/cpu.jpg)

We’re almost done! Now we’re at the middle layer, where everything comes together. Earlier, we talked about how we tell a computer what to do. To make a program like Microsoft Word, we break it down into a set of algorithms, precise and unambiguous instructions that describe exactly what to do in each situation. We use programming languages to write these algorithms in a structured way that the computer can eventually understand.

Then we saw that the computer only understands ones and zeros, so a special program, the compiler, converts our human-readable code into binary machine code.

In the last sections, we saw that we can build circuits that can perform all kinds of operations on binary numbers. Now, we just need to bring those two worlds together and that’s exactly what we’ll do here.

### Assembly and machine code

Previously, we said that the compiler transforms our code into machine code, a list of ones and zeros the computer can execute. Now, we’ve built enough knowledge to start understanding how this works.

An algorithm is just a list of instructions like “if A, then do B.” And A and B might be mathematical operations, comparisons, or decisions. We’ve already seen that we can build circuits that perform those operations. So the only missing piece is this: How does the computer know which operation to do?

That’s what machine code is for. Machine code is a sequence of binary instructions. Each instruction tells the computer:

- What operation to perform (like add, subtract, compare)
- What values to use as input (called operands)

Let’s walk through a basic example. Say you wrote a program that just does 3 + 3. When the compiler processes it, it might generate a machine instruction that looks like this (let’s pretend instructions are 32 bits long):

- The first 8 bits specify the operation. For example, 00000000 could mean "add".
- The next 12 bits encode the first operand.
- The last 12 bits encode the second operand.

So the binary for 3 is 000000000011. The full machine code for 3 + 3 might look like:

$$00000000,000000000011,000000000011$$

Now inside the computer, a circuit reads this instruction, decodes the first 8 bits to know which operation it is (addition in this case), and routes the operands to the right arithmetic circuit (like an adder) to compute the result.

So in short: Every operation in your program is converted into a binary instruction that the CPU can decode and execute.

The exact structure of these instructions is defined by something called the Instruction Set Architecture (ISA) — and it’s actually much more complex than the simple example I gave earlier. In practice, a single line of code in a programming language can be translated into a dozen or more basic machine instructions. The ISA defines what each binary pattern means and what operations are supported. Some computers support thousands of different instructions, each with different formats and behaviors.

This instruction set is essential for both hardware designers (who build the CPU to understand and execute the instructions) and compiler developers (who need to translate high-level code into the correct binary format that the CPU understands).

Some common ISAs you may have heard of include: x86 (Intel, Amd) and ARM (used in smartphones, tablets, and newer Macs)

In the next section, we’ll look briefly at how the CPU and memory work together to execute machine code and manage data.

### CPU and memory.

Now we arrive at the final layer that brings everything together: the CPU/memory architecture. Once a program is compiled, it becomes a list of binary instructions that the computer must process. At its core, a computer consists of two essential components:

- A memory circuit, which stores both the instructions and any intermediate results
- A CPU (Central Processing Unit), which performs the actual computations required by the program

So what happens when you run a program? The machine code is loaded into memory, and the CPU starts executing the instructions, one by one. This process happens in three main steps, repeated over and over:

1. Fetch: The CPU fetches the next instruction from memory
2. Decode: It decodes the instruction to understand what operation is required — addition, subtraction, comparison, etc.
3. Execute: It performs the operation by activating the appropriate circuit

This loop continues for every instruction in the program, from start to finish. Some programs contain loops (so they run indefinitely), or they include wait instructions (for user input or events). But no matter what the program is doing, the CPU always runs in the same cycle: fetch, decode, execute.

Inside the CPU is a special circuit called the ALU (Arithmetic Logic Unit). This unit contains all the circuits we discussed earlier: arithmetic circuits, comparators, multiplexers, and more. When the CPU decodes an instruction, it selects the right part of the ALU to execute it. The decode part is also implemented using logic gates and can correctly select the correct circuit for the execute step.

You may have also heard of clock speed — for example, a 3 GHz CPU. This number tells you how many instructions the CPU can process per second. A 3 GHz processor can perform about 3 billion fetch-decode-execute cycles every second. Each clock tick triggers the next step in the process.


# Conclusion
And with that, we’re done!

If you’ve made it this far, you should be proud of yourself. There was a lot of content, and many layers to follow. This wasn't a light read, but you stuck with it, so congrats!

Let’s quickly recap what we’ve learned: 

We started with sand, and from it, created transistors, tiny electronic switches made from silicon. We then used these transistors to build logic gates, which implement Boolean algebra, a math system based entirely on 0s and 1s.

From there, we combined logic gates to build circuits that can perform meaningful operations like addition, comparison, and control flow. By assembling these circuits together, we built the foundation of a computer: a CPU and memory.

Then we introduced the Instruction Set Architecture (ISA), the set of binary instructions the CPU can understand. We saw how a compiler translates human-readable code into these binary instructions. Finally, we connected everything back to the world of applications, the software we use every day.

Of course, I simplified many concepts and skipped a lot of technical details in each step. But the goal wasn’t to make you an expert, it was to give you a solid understanding of how all the layers of abstraction fit together.

Most engineers spend their careers working at just one of these layers, and they still discover new things every day. So this really is just a starting point, and there’s still a lot more to explore.

Thanks again for reading,

Massil Ait Abdeslam





















