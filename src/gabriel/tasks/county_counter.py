from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd

# Adjust these imports to match your tree
from .elo import EloConfig, EloRater
from .regional import Regional
from ..utils.teleprompter import Teleprompter
from ..utils.maps import create_county_choropleth


class CountyCounter:
    """Run regional analysis on counties (or any regions) and rate them via Elo."""

    def __init__(
        self,
        df: pd.DataFrame,
        county_col: str,
        topics: List[str],
        *,
        fips_col: Optional[str] = None,
        save_dir: str = os.path.expanduser("~/Documents/runs"),
        run_name: str | None = None,
        model_regional: str = "o4-mini",
        model_elo: str = "o4-mini",
        reasoning_effort: str = "medium",
        search_context_size: str = "medium",
        n_parallels: int = 400,
        n_elo_rounds: int = 15,
        elo_timeout: float = 60.0,
        use_dummy: bool = False,
        additional_instructions: str = "",
        elo_instructions: str = "",
        z_score_choropleth: bool = True,
        elo_attributes: Dict[str, str] | None = None,
        print_example_prompt: bool = True,
    ) -> None:
        self.df = df.copy()
        self.county_col = county_col
        self.fips_col = fips_col
        self.topics = topics

        self.model_regional = model_regional
        self.model_elo = model_elo
        self.n_parallels = n_parallels
        self.n_elo_rounds = n_elo_rounds
        self.elo_timeout = elo_timeout
        self.use_dummy = use_dummy

        self.additional_instructions = additional_instructions
        self.elo_instructions = elo_instructions
        self.reasoning_effort = reasoning_effort
        self.search_context_size = search_context_size
        self.z_score_choropleth = z_score_choropleth
        self.elo_attributes = elo_attributes
        self.print_example_prompt = print_example_prompt

        run_name = run_name or f"county_counter_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_path = os.path.join(save_dir, run_name)
        os.makedirs(self.save_path, exist_ok=True)

        # Regional step
        self.regional = Regional(
            df=self.df,
            region_col=self.county_col,
            topics=self.topics,
            save_dir=self.save_path,
            run_name="regional",
            model=self.model_regional,
            n_parallels=self.n_parallels,
            use_dummy=self.use_dummy,
            additional_instructions=self.additional_instructions,
            reasoning_effort=self.reasoning_effort,
            search_context_size=self.search_context_size,
            print_example_prompt=self.print_example_prompt,
        )

        self.tele = Teleprompter(os.path.join(os.path.dirname(__file__), "prompts"))

    async def run(self, *, reset_files: bool = False) -> pd.DataFrame:
        # 1. Generate regional reports
        reports_df = await self.regional.run(reset_files=reset_files)
        results = reports_df[["region"]].copy()
        results["region"] = results["region"].astype(str)

        # 2. For each topic, run Elo and merge results
        for topic in self.topics:
            df_topic = reports_df[["region", topic]].rename(
                columns={"region": "identifier", topic: "text"}
            )

            # Attributes list
            attributes = (
                list(self.elo_attributes.keys()) if self.elo_attributes else [topic]
            )

            cfg = EloConfig(
                attributes=attributes,
                n_rounds=self.n_elo_rounds,
                n_parallels=self.n_parallels,
                model=self.model_elo,
                save_dir=self.save_path,
                run_name=f"elo_{topic}",
                use_dummy=self.use_dummy,
                instructions=self.elo_instructions,
                print_example_prompt=False,
                timeout=self.elo_timeout,
            )

            rater = EloRater(self.tele, cfg)
            elo_df = await rater.run(df_topic, text_col="text", id_col="identifier")
            elo_df["identifier"] = elo_df["identifier"].astype(str)

            if self.elo_attributes:
                for attr in [c for c in elo_df.columns if c not in ("identifier", "text")]:
                    tmp = f"_elo_tmp_{attr}"
                    results = results.merge(
                        elo_df[["identifier", attr]].rename(columns={attr: tmp}),
                        left_on="region",
                        right_on="identifier",
                        how="left",
                    )
                    results[attr] = results[tmp]
                    results = results.drop(columns=["identifier", tmp])
            else:
                tmp = f"_elo_tmp_{topic}"
                results = results.merge(
                    elo_df[["identifier", topic]].rename(columns={topic: tmp}),
                    left_on="region",
                    right_on="identifier",
                    how="left",
                )
                results[topic] = results[tmp]
                results = results.drop(columns=["identifier", tmp])

        # 3. Choropleths (optional)
        if self.fips_col and self.fips_col in self.df.columns:
            merged = (
                self.df[[self.county_col, self.fips_col]]
                .drop_duplicates()
                .copy()
            )
            merged[self.fips_col] = merged[self.fips_col].astype(str).str.zfill(5)
            results = results.merge(
                merged, left_on="region", right_on=self.county_col, how="left"
            )

            if self.elo_attributes:
                for attr in self.elo_attributes.keys():
                    map_path = os.path.join(self.save_path, f"map_{attr}.html")
                    create_county_choropleth(
                        results,
                        fips_col=self.fips_col,
                        value_col=attr,
                        title=f"ELO Rating for {attr}",
                        save_path=map_path,
                        z_score=self.z_score_choropleth,
                    )
            else:
                for topic in self.topics:
                    map_path = os.path.join(self.save_path, f"map_{topic}.html")
                    create_county_choropleth(
                        results,
                        fips_col=self.fips_col,
                        value_col=topic,
                        title=f"ELO Rating for {topic}",
                        save_path=map_path,
                        z_score=self.z_score_choropleth,
                    )

        results.to_csv(os.path.join(self.save_path, "county_elo.csv"), index=False)
        return results
