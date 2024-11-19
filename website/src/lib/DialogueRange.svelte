<script>
	import data from './data/textclusters.json';
	import Wide from './Wide.svelte';

	import * as d3 from 'd3';

	const entropyFmt = d3.format('.2f');
	const colorScale = d3
		.scaleSequential(d3.interpolateRdBu)
		.domain(d3.extent(data, (d) => d.entropy));
</script>

<figure>
	<Wide>
		<div class="table-container">
			<table>
				<thead>
					<tr>
						<th class="text-cell">Text</th>
						<th class="entropy-cell">Entropy</th>
					</tr>
				</thead>
				<tbody>
					{#each data.slice(0, 20) as { id, examples, entropy }}
						<tr style:background-color={colorScale(entropy)}>
							<td class="text-cell"
								>{examples[0]} <span class="extra">{examples.slice(1).join(' ')}</span></td
							>
							<td style:text-align="center" class="entropy-cell">{entropyFmt(entropy)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<table>
				<thead>
					<tr>
						<th class="text-cell">Text</th>
						<th class="entropy-cell">Entropy</th>
					</tr>
				</thead>
				<tbody>
					{#each data.slice(20).reverse() as { id, examples, entropy }}
						<tr style:background-color={colorScale(entropy)}>
							<td class="text-cell"
								>{examples[0]} <span class="extra">{examples.slice(1).join(' ')}</span></td
							>
							<td style:text-align="center" class="entropy-cell">{entropyFmt(entropy)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</Wide>
	<figcaption>
		The dialogue phrases with lowest and highest entropies (the lowest and highest emotional range).
	</figcaption>
</figure>

<style>
	th,
	td {
		font-family: 'Open Sans', sans-serif;
	}
	.text-cell {
		max-width: calc(20vw - 30px);
		white-space: nowrap;
		overflow-x: hidden;
		text-overflow: clip;
	}
	.entropy-cell {
		width: 1em;
	}
	span.extra {
		font-family: 'Open Sans', sans-serif;
		opacity: 0.5;
	}
	th.text-cell {
		text-align: left;
	}
	.table-container {
		display: flex;
		flex-wrap: wrap;
	}

	table {
		width: 50%;
		box-sizing: border-box;
	}

	@media (max-width: 800px) {
		table {
			width: 100%;
		}
		table tbody tr:nth-child(n + 11) {
			display: none;
		}
	}

	figcaption {
		font-size: 0.8em;
		font-style: italic;
	}
	table:nth-child(odd) {
		padding-right: 10px;
	}

	table:nth-child(even) {
		padding-left: 10px;
	}
</style>
