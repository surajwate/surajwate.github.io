const pluginDate = require("eleventy-plugin-date");
const markdownIt = require("markdown-it");
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");

module.exports = function (eleventyConfig) {
  // Copy CNAME file to output directory
  eleventyConfig.addPassthroughCopy("CNAME");
  // Add syntax highlighting
  eleventyConfig.addPlugin(syntaxHighlight);

  // Markdown options
  let exports = function (eleventyConfig) {
    let options = {
      html: true,
      breaks: true,
      linkify: true,
      typographer: true,
    };
    eleventyConfig.setLibrary("md", markdownIt(options));
  }
  // Copy `assets/` to `_site/assets`
  eleventyConfig.addPassthroughCopy("src/assets");
  // Blog collection
  eleventyConfig.addCollection("posts", function (collection) {
    return collection.getFilteredByGlob("src/blog/posts/**/*.md").filter(post => !post.data.draft);
  });
  // Projects collection
  eleventyConfig.addCollection("projects", function (collection) {
    return collection.getFilteredByGlob("src/projects/**/*.md").filter(project => !project.data.draft);
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