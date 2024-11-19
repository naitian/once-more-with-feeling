<script>
	import Wide from './Wide.svelte';
    import { base } from '$app/paths';
</script>

<figure>
	<Wide>
		<img class="wide" src={`${base}/wide.png`} alt="A wide shot of Audrey Hepburn from the film Charade." />
		<img
			class="close"
			src={`${base}/close.png`}
			alt="A close-up of Audrey Hepburn showing the details of her expression from the film Charade."
		/>
	</Wide>
	<figcaption>
		The close-up shot of Audrey Hepburn in <i>Charade</i> (1963) reveals details that are invisible from
		a distance.
	</figcaption>
</figure>

<style>
	img {
		width: 45%;
		padding: 0 2%;
		aspect-ratio: 1 / 1;
		object-fit: cover;
		display: inline-block;
	}
	figcaption {
		font-size: 0.8em;
		font-style: italic;
	}
</style>
