<script>
	import { tick } from 'svelte';
	import { onMount } from 'svelte';
	import scrollama from 'scrollama';
	import Wide from './Wide.svelte';
	import * as d3 from 'd3';
	import { tweened } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';
	import AnimatedLine from './AnimatedLine.svelte';

	import joyData from './data/joy_trajectory.json';
	import sadnessData from './data/sadness_trajectory.json';

	const neutralData = [
		[0.0, 0.4310574268610108],
		[0.5, 0.41630263889743685],
		[1.0, 0.4185491666096792],
		[1.5, 0.42758650351173605],
		[2.0, 0.43193307629721145],
		[2.5, 0.4335089983228245],
		[3.0, 0.4418159777610866],
		[3.5, 0.4473989366873576],
		[4.0, 0.4499927338902544],
		[4.5, 0.45213147470899895],
		[5.0, 0.4545039320881394],
		[5.5, 0.4600362782576361],
		[6.0, 0.4592063774119357],
		[6.5, 0.4660309569373845],
		[7.0, 0.4738437517318147],
		[7.5, 0.47676304710537903],
		[8.0, 0.48435965092254823],
		[8.5, 0.4965258529618846],
		[9.0, 0.48418837996488984],
		[9.5, 0.49183428029664183]
	];

	let margin = { top: 0, right: 20, bottom: 40, left: 45 };
	let totalWidth = 0;
	let totalHeight = 0;
	let lines = [];
	$: width = totalWidth - margin.left - margin.right;
	$: height = totalHeight - margin.top - margin.bottom;

	let xDomain = [0, 10];
	let yDomain = [0, 1];
	$: xScale = d3.scaleLinear().domain(xDomain).range([0, width]);
	$: yScale = d3.scaleLinear().domain(yDomain).range([height, 0]);

	let xAxis = d3.axisBottom(xScale).ticks(10);
	let yAxis = d3.axisLeft(yScale).ticks(5);

	const tickFormat = d3.format('.0%');

	const line = d3
		.line()
		.curve(d3.curveCatmullRom.alpha(0.1))
		.x((d) => xScale(d[0] + 0.5))
		.y((d) => yScale(d[1]));

    let color = 'white';

	onMount(() => {
		const scroller = scrollama();
		scroller
			.setup({
				step: '.step',
				offset: 0.5,
				progress: true
			})
			.onStepEnter(({ index, element, direction }) => {
				if (index === 0) {
					yDomain = [0.4, 0.5];
                    color = "white";
					lines = [neutralData];
				} else if (index === 1) {
					lines = [];
				} else if (index === 2) {
					yDomain = [0.4, 0.5];
					lines = [joyData];
                    color = "#FADA5E";
				} else if (index === 3) {
					yDomain = [0.1, 0.18];
					lines = [sadnessData];
                    color = "skyblue"
				} else {
					lines = [];
				}
			});
	});
</script>

<div class="container">
	<div class="graphic">
		<Wide>
			<svg bind:clientWidth={totalWidth} bind:clientHeight={totalHeight}>
				<g class="figure" transform={`translate(${margin.left} ${margin.top})`}>
					{#each lines as data}
						{#key data}
							<AnimatedLine d={line(data)} stroke={color}/>
						{/key}
					{/each}
					<rect {width} {height} fill="none"></rect>
				</g>
				<g class="x-axis axis" transform={`translate(${margin.left} ${margin.top + height})`}>
					<g class="x-ticks ticks">
						{#each xScale.ticks(10).slice(1) as tick}
							<g transform={`translate(${xScale(tick)} 0)`}>
								<text text-anchor="middle" dominant-baseline="hanging">{tickFormat(tick / 10)}</text
								>
							</g>
						{/each}
					</g>
					<text transform={`translate(${width / 2} 30)`} text-anchor="middle" fill="white"
						>Narrative Time</text
					>
				</g>
				<g class="y-axis axis" transform={`translate(${margin.left} 0)`}>
					<g class="y-ticks ticks">
						{#each yScale.ticks(10) as tick}
							<g transform={`translate(-2 ${yScale(tick)})`}>
								<text text-anchor="end">{tickFormat(tick)}</text>
							</g>
						{/each}
					</g>
					<text
						transform={`translate(${-30} ${height / 2}) rotate(-90)`}
						text-anchor="middle"
						fill="white">Proportion of utterances</text
					>
				</g>
			</svg>
		</Wide>
	</div>

	<div class="step">
		<p>
			We first find that the proportion of utterances that are associated with <b>any</b> emotion steadily
			increases over the course of the film.
		</p>
	</div>
	<div class="step">
		<p>But when we look at specific emotions, we find non-linear trajectories.</p>
	</div>
	<div class="step">
		<p>Like <b style:color="#FADA5E">joy</b>, which tends to be highest at the beginning and end.</p>
	</div>
	<div class="step"><p>And <b style:color="skyblue">sadness</b>, which has roughly the opposite shape.</p></div>
</div>

<style>
	.container {
		position: relative;
	}

	.graphic {
		position: sticky;
		top: 25vh;
		width: 100%;
		height: 50vh;
	}

	.graphic svg {
		width: 100%;
		height: 50vh;
	}
	svg text {
		font-family: 'Open Sans', sans-serif;
		fill: white;
	}

	.axis text {
		fill: #fff8;
	}
	.ticks text {
		font-size: 0.7em;
	}

	.step {
		width: 80%;
		padding: 0 10%;
		height: 90vh;
		position: relative;
		z-index: 10;
		display: flex;
		justify-content: center;
		align-items: center;
	}
    .step p {
        background-color: #111;
    }
</style>
