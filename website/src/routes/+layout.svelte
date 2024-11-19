<script>
	import { dev } from '$app/environment';
	import { on } from 'svelte/events';
</script>

<svelte:head>
	<meta http-equiv="content-type" content="text/html; charset=utf-8" />
	<meta http-equiv="X-UA-Compatible" content="IE=edge" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Once More, With Feeling</title>

	{#if !dev}
		<!-- Global site tag (gtag.js) - Google Analytics -->
		<script async src="https://www.googletagmanager.com/gtag/js?id=UA-89717266-1"></script>
		<script>
			window.dataLayer = window.dataLayer || [];
			function gtag() {
				dataLayer.push(arguments);
			}
			gtag('js', new Date());
			gtag('config', 'UA-89717266-1');
		</script>
	{/if}

	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
	<link
		href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;0,700;1,400;1,700&display=swap"
		rel="stylesheet"
	/>
	<link
		href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<slot></slot>

<style global>
	* {
		font-family: Merriweather;
	}

	body {
		margin: 0;
		padding: 0;
		min-height: 100vh;
		line-height: 1.7;
        scroll-behavior: smooth!important;
	}

    a {
        color: inherit;
    }

	/* .content {
		max-width: 600px;
		width: 80vw;
		margin: 3rem auto;
	} */

	h1,
	h2 {
		font-weight: 400;
	}
</style>
