export default class LocalStorage {
    static save(storageKey, data) {
        return browser.storage.local.set({ [storageKey]: data }).then(() => {
            console.log("Saving " + storageKey + " successful");
        });
    }

    static load(storageKey) {
        return browser.storage.local.get(storageKey).then((results) => {
            if (results[storageKey] != null) {
                console.log("Loading " + storageKey + " successful");
                return results[storageKey];
            } else {
                throw("No saved data found for this key: " + storageKey);
            }
        });
    }

    static delete(storageKey) {
        return browser.storage.local.remove(storageKey).then(() => {
            console.log("Removing of " + storageKey + " successful");
        });
    }

    static clear() {
        return browser.storage.local.clear();
    }
}