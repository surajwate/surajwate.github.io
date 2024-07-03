const pluginDate = require("eleventy-plugin-date");

module.exports = function(eleventyConfig) {
    // Copy `assets/` to `_site/assets`
    eleventyConfig.addPassthroughCopy("src/assets");
    // Blog collection
    eleventyConfig.addCollection("posts", function(collection) {
      return collection.getFilteredByGlob("src/blog/posts/**/*.md");
    });

    eleventyConfig.addPlugin(pluginDate, {
        // Specify custom date formats
        formats: {
        // Change the readableDate filter to use abbreviated months.
        readableDate: { year: "numeric", month: "short", day: "numeric" },
        // Add a new filter to format a Date to just the month and year.
        readableMonth: { year: "numeric", month: "long" },
        // Add a new filter using formatting tokens.
        timeZone: "z",
        }
    });

    return {
      dir: {
        input: "src",
        output: "_site",
        includes: "_includes",
      },
      templateFormats: ["html", "md", "njk"],
      htmlTemplateEngine: "njk",
      markdownTemplateEngine: "njk",
      dataTemplateEngine: "njk",
    };
  }