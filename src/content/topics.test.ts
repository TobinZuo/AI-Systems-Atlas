import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { availableTopicCount, knowledgeDomains, learningJourneys, topicCount, topicIndex } from "./topics";

describe("Knowledge topic catalog", () => {
  it("keeps domains and topics unique", () => {
    const domainIds = knowledgeDomains.map((domain) => domain.id);
    const topicIds = knowledgeDomains.flatMap((domain) => domain.topics.map((topic) => topic.id));

    assert.equal(new Set(domainIds).size, domainIds.length);
    assert.equal(new Set(topicIds).size, topicIds.length);
    assert.equal(topicIndex.size, topicCount);
  });

  it("only references known topics", () => {
    for (const domain of knowledgeDomains) {
      for (const topic of domain.topics) {
        for (const prerequisite of topic.prerequisites ?? []) {
          assert.ok(topicIndex.has(prerequisite), `${topic.id} references missing ${prerequisite}`);
        }
      }
    }

    for (const journey of learningJourneys) {
      for (const topicId of journey.topicIds) {
        assert.ok(topicIndex.has(topicId), `${journey.id} references missing ${topicId}`);
      }
    }
  });

  it("gives every available topic a route", () => {
    for (const topic of topicIndex.values()) {
      if (topic.status === "available") assert.ok(topic.route);
    }
    assert.equal(availableTopicCount, 12);
  });
});
