using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace ZakuskaAI.Migrations
{
    /// <inheritdoc />
    public partial class historyadded : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "SearchHistory",
                table: "Users",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "SearchHistory",
                table: "Users");
        }
    }
}
