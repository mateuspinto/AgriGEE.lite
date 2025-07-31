from textual.app import App, ComposeResult
from textual.timer import Timer
from textual_pandas.widgets import DataFrameTable

import agrigee_lite as agl


class GuiTasksApp(App):
    def compose(self) -> ComposeResult:
        yield DataFrameTable()

    def on_mount(self) -> None:
        self.table = self.query_one(DataFrameTable)

        self.table.cursor_type = "row"
        self.table.zebra_stripes = True
        self.table.fixed_columns = 1
        self.table.show_header = True

        self.refresh_table()
        self.timer: Timer = self.set_interval(30, self.refresh_table)

    def refresh_table(self) -> None:
        df = agl.get_all_tasks()
        self.table.update_df(df)


def main():
    app = GuiTasksApp()
    app.run()


if __name__ == "__main__":
    app = GuiTasksApp()
    app.run()
