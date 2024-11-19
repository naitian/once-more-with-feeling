<script>
	import { onMount } from 'svelte';

	export let d;
	export let duration = 500;
	export let stroke = 'white';

	import { cubicOut } from 'svelte/easing';
	import { tweened } from 'svelte/motion';

	let pathLength = tweened(1, { duration, easing: cubicOut });
	onMount(() => {
		pathLength.set(100);
	});
</script>

<path
	{d}
	fill="none"
	{stroke}
    stroke-width="2"
	pathLength="100"
	stroke-dasharray={`${$pathLength} ${100 - $pathLength}`}
></path>
